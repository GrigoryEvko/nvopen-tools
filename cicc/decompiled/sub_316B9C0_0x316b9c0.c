// Function: sub_316B9C0
// Address: 0x316b9c0
//
__int64 __fastcall sub_316B9C0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r15
  int v4; // r13d
  __int64 v5; // r14
  __int64 v6; // rcx
  char v7; // al
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rsi
  bool v13; // cc
  unsigned __int64 v14; // rdi
  __int64 v16; // [rsp+0h] [rbp-40h]
  char v17; // [rsp+Fh] [rbp-31h]

  ++a1[1];
  v3 = a1[2];
  v4 = *((_DWORD *)a1 + 8);
  v5 = a1[3];
  a1[2] = 0;
  v6 = *a1;
  v7 = *((_BYTE *)a1 + 40);
  a1[3] = 0;
  *((_DWORD *)a1 + 8) = 0;
  v16 = v6;
  *a1 = *a2;
  v17 = v7;
  sub_C7D6A0(0, 0, 8);
  *((_DWORD *)a1 + 8) = 0;
  a1[3] = 0;
  a1[2] = 0;
  ++a1[1];
  v8 = a2[2];
  ++a2[1];
  v9 = a1[2];
  a1[2] = v8;
  LODWORD(v8) = *((_DWORD *)a2 + 6);
  a2[2] = v9;
  LODWORD(v9) = *((_DWORD *)a1 + 6);
  *((_DWORD *)a1 + 6) = v8;
  LODWORD(v8) = *((_DWORD *)a2 + 7);
  *((_DWORD *)a2 + 6) = v9;
  LODWORD(v9) = *((_DWORD *)a1 + 7);
  *((_DWORD *)a1 + 7) = v8;
  LODWORD(v8) = *((_DWORD *)a2 + 8);
  *((_DWORD *)a2 + 7) = v9;
  LODWORD(v9) = *((_DWORD *)a1 + 8);
  *((_DWORD *)a1 + 8) = v8;
  *((_DWORD *)a2 + 8) = v9;
  *((_BYTE *)a1 + 40) = *((_BYTE *)a2 + 40);
  v10 = *((unsigned int *)a2 + 8);
  *a2 = v16;
  if ( (_DWORD)v10 )
  {
    v11 = a2[2];
    v12 = v11 + 32 * v10;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v11 != -4096 && *(_QWORD *)v11 != -8192 )
        {
          if ( *(_BYTE *)(v11 + 24) )
          {
            v13 = *(_DWORD *)(v11 + 16) <= 0x40u;
            *(_BYTE *)(v11 + 24) = 0;
            if ( !v13 )
            {
              v14 = *(_QWORD *)(v11 + 8);
              if ( v14 )
                break;
            }
          }
        }
        v11 += 32;
        if ( v12 == v11 )
          goto LABEL_10;
      }
      v11 += 32;
      j_j___libc_free_0_0(v14);
    }
    while ( v12 != v11 );
LABEL_10:
    v10 = *((unsigned int *)a2 + 8);
  }
  sub_C7D6A0(a2[2], 32 * v10, 8);
  a2[2] = v3;
  ++a2[1];
  a2[3] = v5;
  *((_DWORD *)a2 + 8) = v4;
  *((_BYTE *)a2 + 40) = v17;
  return sub_C7D6A0(0, 0, 8);
}
