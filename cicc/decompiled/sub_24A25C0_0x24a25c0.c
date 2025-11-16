// Function: sub_24A25C0
// Address: 0x24a25c0
//
char __fastcall sub_24A25C0(__int64 a1, __int64 a2, __int64 *a3)
{
  _DWORD *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  char result; // al
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 i; // r13
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+18h] [rbp-58h]
  unsigned __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  __int64 v19; // [rsp+28h] [rbp-48h]
  char v20; // [rsp+30h] [rbp-40h]

  if ( !a2 || !*(_QWORD *)(a1 + 8) )
    return 0;
  sub_B2EE70((__int64)&v18, a2, 0);
  if ( !v20 || (result = sub_D84450(a1, v18)) != 0 )
  {
    v4 = *(_DWORD **)(a1 + 8);
    v17 = a2 + 72;
    if ( !v4 || *v4 != 2 )
      goto LABEL_6;
    v16 = 0;
    v10 = *(_QWORD *)(a2 + 80);
    if ( v10 != a2 + 72 )
    {
      do
      {
        if ( !v10 )
          BUG();
        v11 = *(_QWORD *)(v10 + 32);
        for ( i = v10 + 24; i != v11; v16 += v18 )
        {
          while ( 1 )
          {
            if ( !v11 )
              BUG();
            v13 = *(_BYTE *)(v11 - 24);
            if ( v13 == 34 || v13 == 85 )
            {
              v14 = sub_D84370(a1, v11 - 24, 0, 0);
              v19 = v15;
              v18 = v14;
              if ( (_BYTE)v15 )
                break;
            }
            v11 = *(_QWORD *)(v11 + 8);
            if ( i == v11 )
              goto LABEL_27;
          }
          v11 = *(_QWORD *)(v11 + 8);
        }
LABEL_27:
        v10 = *(_QWORD *)(v10 + 8);
      }
      while ( v10 != v17 );
    }
    result = sub_D84450(a1, v16);
    if ( result )
    {
LABEL_6:
      v5 = *(_QWORD *)(a2 + 80);
      if ( v5 == v17 )
        return 1;
      while ( 1 )
      {
        v6 = v5 - 24;
        if ( !v5 )
          v6 = 0;
        v7 = sub_FDD2C0(a3, v6, 0);
        v19 = v8;
        v18 = v7;
        if ( !(_BYTE)v8 || !sub_D84450(a1, v18) )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 == v17 )
          return 1;
      }
      return 0;
    }
  }
  return result;
}
