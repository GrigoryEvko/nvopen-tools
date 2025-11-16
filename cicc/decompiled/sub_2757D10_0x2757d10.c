// Function: sub_2757D10
// Address: 0x2757d10
//
__int64 __fastcall sub_2757D10(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  char v10; // dl
  __int64 v11; // rdi
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // r14
  __int64 v15; // r15
  unsigned __int64 v16; // rdi
  int v17; // r13d
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  unsigned __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v20 = a1 + 16;
  v19 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v22, a6);
  v7 = v19;
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      while ( 1 )
      {
        if ( v7 )
        {
          *(_DWORD *)v7 = *(_DWORD *)v8;
          v10 = *(_BYTE *)(v8 + 4);
          *(_DWORD *)(v7 + 16) = 0;
          *(_BYTE *)(v7 + 4) = v10;
          *(_QWORD *)(v7 + 8) = v7 + 24;
          *(_DWORD *)(v7 + 20) = 2;
          if ( *(_DWORD *)(v8 + 16) )
            break;
        }
        v8 += 88LL;
        v7 += 88;
        if ( v9 == v8 )
          goto LABEL_7;
      }
      v11 = v7 + 8;
      v21 = v8;
      v7 += 88;
      sub_27578C0(v11, (__int64 *)(v8 + 8));
      v8 = v21 + 88;
    }
    while ( v9 != v21 + 88 );
LABEL_7:
    v12 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v13 = *(unsigned int *)(v9 - 72);
        v14 = *(_QWORD *)(v9 - 80);
        v9 -= 88LL;
        v13 *= 32;
        v15 = v14 + v13;
        if ( v14 != v14 + v13 )
        {
          do
          {
            v15 -= 32;
            if ( *(_DWORD *)(v15 + 24) > 0x40u )
            {
              v16 = *(_QWORD *)(v15 + 16);
              if ( v16 )
                j_j___libc_free_0_0(v16);
            }
            if ( *(_DWORD *)(v15 + 8) > 0x40u && *(_QWORD *)v15 )
              j_j___libc_free_0_0(*(_QWORD *)v15);
          }
          while ( v14 != v15 );
          v14 = *(_QWORD *)(v9 + 8);
        }
        if ( v14 != v9 + 24 )
          _libc_free(v14);
      }
      while ( v12 != v9 );
      v9 = *(_QWORD *)a1;
    }
  }
  v17 = v22[0];
  if ( v20 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v17;
  *(_QWORD *)a1 = v19;
  return v19;
}
