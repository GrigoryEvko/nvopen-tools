// Function: sub_130CDD0
// Address: 0x130cdd0
//
__int64 sub_130CDD0()
{
  unsigned __int64 v0; // rax
  unsigned int v1; // eax
  unsigned int v2; // r12d
  __int64 v3; // rbx
  void *v4; // rax
  void *v5; // r12
  _QWORD *v7; // r12
  void *v8; // rbx
  _BYTE v9[17]; // [rsp+Fh] [rbp-11h] BYREF

  v0 = sysconf(30);
  if ( v0 == -1 )
  {
    qword_4F969C8 = 12;
  }
  else
  {
    qword_4F969C8 = v0;
    if ( v0 > 0x1000 )
    {
      sub_130AA40("<jemalloc>: Unsupported system page size\n");
      if ( !byte_4F969A5[0] )
        return 1;
      goto LABEL_32;
    }
  }
  if ( unk_4F969A0 )
  {
LABEL_4:
    dword_4C6F0F0 = 0;
    goto LABEL_5;
  }
  v7 = mmap(0, 0x1000u, 3, 34, -1, 0);
  if ( v7 == (_QWORD *)-1LL )
  {
    sub_130AA40("<jemalloc>: Cannot allocate memory for MADV_DONTNEED check\n");
    if ( byte_4F969A5[0] )
LABEL_32:
      abort();
  }
  *v7 = 0x4141414141414141LL;
  v7[511] = 0x4141414141414141LL;
  memset(
    (void *)((unsigned __int64)(v7 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    0x41u,
    8LL * (((unsigned int)v7 - (((_DWORD)v7 + 8) & 0xFFFFFFF8) + 4096) >> 3));
  if ( madvise(v7, 0x1000u, 4) )
  {
    if ( !munmap(v7, 0x1000u) )
      goto LABEL_4;
    sub_130AA40("<jemalloc>: Cannot deallocate memory for MADV_DONTNEED check\n");
    if ( !byte_4F969A5[0] )
      goto LABEL_4;
    goto LABEL_32;
  }
  v8 = memchr(v7, 65, 0x1000u);
  if ( munmap(v7, 0x1000u) )
  {
    sub_130AA40("<jemalloc>: Cannot deallocate memory for MADV_DONTNEED check\n");
    if ( byte_4F969A5[0] )
      goto LABEL_32;
  }
  dword_4C6F0F0 = v8 != 0;
  if ( v8 )
  {
    sub_130AA40("<jemalloc>: MADV_DONTNEED does not work (memset will be used instead)\n");
    sub_130AA40("<jemalloc>: (This is the expected behaviour if you are running under QEMU)\n");
  }
LABEL_5:
  flags = 34;
  v1 = syscall(2, "/proc/sys/vm/overcommit_memory", 0x80000);
  v2 = v1;
  if ( v1 == -1 || (v3 = syscall(0, v1, v9, 1), syscall(3, v2), v3 <= 0) )
  {
    byte_4F969C0 = 0;
  }
  else
  {
    byte_4F969C0 = (unsigned __int8)(v9[0] - 48) <= 1u;
    if ( (unsigned __int8)(v9[0] - 48) <= 1u )
      flags |= 0x4000u;
  }
  if ( unk_4F96B94 && byte_4F969A5[0] )
  {
    sub_130AA40("<jemalloc>: no MADV_HUGEPAGE support\n");
    abort();
  }
  v9[0] = 0;
  unk_505F9C8 = 3;
  unk_4F969BC = 3;
  v4 = sub_130C9C0(0, 0x1000u, v9);
  v5 = v4;
  if ( !v4 )
    return 1;
  if ( sub_130CD50(v4, 0x1000u) )
    byte_4C6F0F4 = 0;
  sub_130C960(v5, 0x1000u);
  return 0;
}
