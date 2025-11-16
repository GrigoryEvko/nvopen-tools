// Function: sub_13C9E40
// Address: 0x13c9e40
//
void __fastcall sub_13C9E40(__int64 a1)
{
  void *v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rdx
  unsigned __int64 *v5; // r15
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rax

  ++*(_QWORD *)(a1 + 40);
  v2 = *(void **)(a1 + 56);
  if ( v2 == *(void **)(a1 + 48) )
    goto LABEL_6;
  v3 = 4 * (*(_DWORD *)(a1 + 68) - *(_DWORD *)(a1 + 72));
  v4 = *(unsigned int *)(a1 + 64);
  if ( v3 < 0x20 )
    v3 = 32;
  if ( (unsigned int)v4 <= v3 )
  {
    memset(v2, -1, 8 * v4);
LABEL_6:
    *(_QWORD *)(a1 + 68) = 0;
    goto LABEL_7;
  }
  sub_16CC920(a1 + 40);
LABEL_7:
  v5 = *(unsigned __int64 **)(a1 + 216);
  v6 = (unsigned __int64 *)(a1 + 208);
  while ( v6 != v5 )
  {
    v7 = v5;
    v5 = (unsigned __int64 *)v5[1];
    v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
    *v5 = v8 | *v5 & 7;
    *(_QWORD *)(v8 + 8) = v5;
    v9 = v7[8];
    *v7 &= 7u;
    v7[1] = 0;
    *(v7 - 4) = (unsigned __int64)&unk_49EA628;
    if ( v9 != v7[7] )
      _libc_free(v9);
    v10 = v7[5];
    if ( v10 != 0 && v10 != -8 && v10 != -16 )
      sub_1649B30(v7 + 3);
    *(v7 - 4) = (unsigned __int64)&unk_49EE2B0;
    v11 = *(v7 - 1);
    if ( v11 != -8 && v11 != 0 && v11 != -16 )
      sub_1649B30(v7 - 3);
    j_j___libc_free_0(v7 - 4, 136);
  }
}
