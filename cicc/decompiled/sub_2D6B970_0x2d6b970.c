// Function: sub_2D6B970
// Address: 0x2d6b970
//
__int64 __fastcall sub_2D6B970(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // r15
  __int64 v14; // rdi
  unsigned __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rax
  char **v18; // rsi
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rdi
  int v21; // r15d
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v24, a6);
  v13 = *(_QWORD *)a1;
  v23 = v8;
  v14 = v8;
  v15 = v13 + 24LL * *(unsigned int *)(a1 + 8);
  if ( v13 != v15 )
  {
    do
    {
      while ( 1 )
      {
        v16 = 24;
        if ( v14 )
        {
          v17 = *(_QWORD *)v13;
          v16 = v14 + 24;
          *(_DWORD *)(v14 + 16) = 0;
          *(_QWORD *)(v14 + 8) = v14 + 24;
          *(_QWORD *)v14 = v17;
          *(_DWORD *)(v14 + 20) = 0;
          if ( *(_DWORD *)(v13 + 16) )
            break;
        }
        v13 += 24LL;
        v14 = v16;
        if ( v15 == v13 )
          goto LABEL_7;
      }
      v18 = (char **)(v13 + 8);
      v13 += 24LL;
      sub_2D56E90(v14 + 8, v18, v9, v10, v11, v12);
      v14 += 24;
    }
    while ( v15 != v13 );
LABEL_7:
    v19 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 24LL;
        v20 = *(_QWORD *)(v15 + 8);
        if ( v20 != v15 + 24 )
          _libc_free(v20);
      }
      while ( v15 != v19 );
      v15 = *(_QWORD *)a1;
    }
  }
  v21 = v24[0];
  if ( v6 != v15 )
    _libc_free(v15);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return v23;
}
