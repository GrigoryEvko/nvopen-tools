// Function: sub_30E2CB0
// Address: 0x30e2cb0
//
__int64 *__fastcall sub_30E2CB0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v9; // eax
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // eax
  void *v13; // rdx
  __int64 v14; // rax
  int v16; // r8d

  v9 = *(_DWORD *)(a4 + 24);
  v10 = *(_QWORD *)(a4 + 8);
  if ( !v9 )
  {
LABEL_7:
    sub_30E2940(a1, a2, a3);
    return a1;
  }
  v11 = v9 - 1;
  v12 = (v9 - 1) & (((unsigned int)&unk_5030EC8 >> 9) ^ ((unsigned int)&unk_5030EC8 >> 4));
  v13 = *(void **)(v10 + 16LL * v12);
  if ( v13 != &unk_5030EC8 )
  {
    v16 = 1;
    while ( v13 != (void *)-4096LL )
    {
      v12 = v11 & (v16 + v12);
      v13 = *(void **)(v10 + 16LL * v12);
      if ( v13 == &unk_5030EC8 )
        goto LABEL_3;
      ++v16;
    }
    goto LABEL_7;
  }
LABEL_3:
  v14 = sub_BC0510(a4, &unk_5030EC8, a5);
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(v14 + 8))(a1, a2, a3, a4, a5);
  return a1;
}
