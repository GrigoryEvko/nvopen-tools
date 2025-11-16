// Function: sub_2B49530
// Address: 0x2b49530
//
__int64 __fastcall sub_2B49530(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  int v19; // r12d
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v22 = a1 + 16;
  v21 = sub_C8D7D0(a1, a1 + 16, a2, 0x118u, v23, a6);
  v7 = v21;
  v8 = *(_QWORD *)a1 + 280LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = *(_QWORD *)a1;
    do
    {
      if ( v7 )
      {
        v10 = *(_QWORD *)v9;
        *(_QWORD *)(v7 + 8) = 0;
        *(_DWORD *)(v7 + 16) = 1;
        *(_QWORD *)v7 = v10;
        *(_DWORD *)(v7 + 20) = 0;
        if ( v7 == -24 || (*(_QWORD *)(v7 + 24) = -1, v7 != -40) )
          *(_QWORD *)(v7 + 40) = -1;
        sub_2B48B90(v7 + 8, v9 + 8);
        *(_DWORD *)(v7 + 64) = 0;
        *(_QWORD *)(v7 + 56) = v7 + 72;
        *(_DWORD *)(v7 + 68) = 2;
        if ( *(_DWORD *)(v9 + 64) )
          sub_2B49010(v7 + 56, (__int64 *)(v9 + 56), v7 + 72, v11, v12, v13);
      }
      v9 += 280LL;
      v7 += 280;
    }
    while ( v8 != v9 );
    v14 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 280LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v15 = *(unsigned int *)(v8 - 216);
        v16 = *(_QWORD *)(v8 - 224);
        v8 -= 280LL;
        v17 = v16 + 104 * v15;
        if ( v16 != v17 )
        {
          do
          {
            v17 -= 104LL;
            v18 = *(_QWORD *)(v17 + 56);
            if ( v18 != v17 + 72 )
              _libc_free(v18);
            if ( (*(_BYTE *)(v17 + 16) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v17 + 24), 16LL * *(unsigned int *)(v17 + 32), 8);
          }
          while ( v16 != v17 );
          v16 = *(_QWORD *)(v8 + 56);
        }
        if ( v16 != v8 + 72 )
          _libc_free(v16);
        if ( (*(_BYTE *)(v8 + 16) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v8 + 24), 16LL * *(unsigned int *)(v8 + 32), 8);
      }
      while ( v8 != v14 );
      v8 = *(_QWORD *)a1;
    }
  }
  v19 = v23[0];
  if ( v22 != v8 )
    _libc_free(v8);
  *(_DWORD *)(a1 + 12) = v19;
  *(_QWORD *)a1 = v21;
  return v21;
}
