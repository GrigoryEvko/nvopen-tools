// Function: sub_25DC2E0
// Address: 0x25dc2e0
//
__int64 __fastcall sub_25DC2E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int8 *v6; // rax
  unsigned __int8 *v7; // r15
  __int64 v8; // r15
  __int64 v9; // r14
  _BYTE *v10; // r15
  __int64 v12; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 16);
  while ( v3 )
  {
    while ( 1 )
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      v5 = *(_QWORD *)(v4 + 24);
      if ( *(_BYTE *)v5 == 85 )
      {
        v6 = sub_BD3990(*(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)), a2);
        v7 = v6;
        if ( !*v6 && !sub_B2FC80((__int64)v6) )
        {
          v8 = *((_QWORD *)v7 + 10);
          if ( !v8 )
            BUG();
          v9 = *(_QWORD *)(v8 + 32);
          v12 = v8 + 24;
          if ( v9 != v8 + 24 )
          {
            while ( 1 )
            {
              v10 = 0;
              if ( v9 )
                v10 = (_BYTE *)(v9 - 24);
              if ( !sub_B46AA0((__int64)v10) )
                break;
              v9 = *(_QWORD *)(v9 + 8);
              if ( v12 == v9 )
                goto LABEL_3;
            }
            if ( *v10 == 30 )
              break;
          }
        }
      }
LABEL_3:
      if ( !v3 )
        return v2;
    }
    v2 = 1;
    a2 = sub_AD6530(*(_QWORD *)(v5 + 8), a2);
    sub_BD84D0(v5, a2);
    sub_B43D60((_QWORD *)v5);
  }
  return v2;
}
