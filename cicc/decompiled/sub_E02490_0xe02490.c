// Function: sub_E02490
// Address: 0xe02490
//
void __fastcall sub_E02490(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int8 *v12; // rax
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v4 = a3;
  v6 = *(_QWORD *)(a3 + 16);
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 72LL) + 40LL);
  v8 = a2[2];
  if ( v6 )
  {
    v9 = (__int64)a2;
    a2 += 4;
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v6 + 24);
        if ( *(_BYTE *)v10 == 85 )
        {
          v11 = *(_QWORD *)(v10 - 32);
          if ( v11 )
          {
            if ( !*(_BYTE *)v11
              && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v10 + 80)
              && (*(_BYTE *)(v11 + 33) & 0x20) != 0
              && *(_DWORD *)(v11 + 36) == 11 )
            {
              break;
            }
          }
        }
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_13;
      }
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 12) )
      {
        v13 = a4;
        v15 = v4;
        sub_C8D5F0(v9, a2, v8 + 1, 8u, v4, a4);
        v8 = *(unsigned int *)(v9 + 8);
        a4 = v13;
        v4 = v15;
      }
      *(_QWORD *)(*(_QWORD *)v9 + 8 * v8) = v10;
      v8 = (unsigned int)(*(_DWORD *)(v9 + 8) + 1);
      *(_DWORD *)(v9 + 8) = v8;
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
  }
LABEL_13:
  v16 = a4;
  if ( (_DWORD)v8 )
  {
    v14 = v4;
    v12 = sub_BD3990(*(unsigned __int8 **)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)), (__int64)a2);
    sub_E02190(v7, a1, (__int64)v12, 0, v14, v16);
  }
}
