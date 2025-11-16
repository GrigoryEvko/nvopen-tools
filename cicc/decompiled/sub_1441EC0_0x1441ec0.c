// Function: sub_1441EC0
// Address: 0x1441ec0
//
__int64 __fastcall sub_1441EC0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v4; // rsi
  int v5; // edx
  unsigned __int8 v6; // al
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 j; // r13
  char v12; // dl
  unsigned __int64 i; // [rsp+10h] [rbp-60h]
  unsigned __int8 v15; // [rsp+1Fh] [rbp-51h]
  __int64 v17; // [rsp+28h] [rbp-48h]
  __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  char v19; // [rsp+38h] [rbp-38h]

  if ( !a2 )
    return 0;
  v15 = sub_1441AE0((_QWORD *)a1);
  if ( !v15 )
    return 0;
  v4 = sub_15E44B0(a2);
  if ( v5 )
  {
    v6 = sub_1441CD0(a1, v4);
    if ( v6 )
      return v6;
  }
  v17 = a2 + 72;
  if ( (unsigned __int8)sub_1441AE0((_QWORD *)a1) )
  {
    v9 = *(_QWORD *)(a2 + 80);
    v7 = v9;
    if ( **(_DWORD **)(a1 + 8) == 1 )
    {
      for ( i = 0; v9 != v17; v9 = *(_QWORD *)(v9 + 8) )
      {
        if ( !v9 )
          BUG();
        v10 = *(_QWORD *)(v9 + 24);
        for ( j = v9 + 16; j != v10; i += v18 )
        {
          while ( 1 )
          {
            if ( !v10 )
              BUG();
            v12 = *(_BYTE *)(v10 - 8);
            if ( v12 == 78 || v12 == 29 )
            {
              sub_1441B50((__int64)&v18, a1, v10 - 24, 0);
              if ( v19 )
                break;
            }
            v10 = *(_QWORD *)(v10 + 8);
            if ( j == v10 )
              goto LABEL_27;
          }
          v10 = *(_QWORD *)(v10 + 8);
        }
LABEL_27:
        ;
      }
      if ( sub_1441CD0(a1, i) )
        return v15;
      v7 = *(_QWORD *)(a2 + 80);
    }
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 80);
  }
  if ( v7 == v17 )
    return 0;
  while ( 1 )
  {
    v8 = v7 - 24;
    if ( !v7 )
      v8 = 0;
    if ( sub_1441E70(a1, v8, a3) )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( v7 == v17 )
      return 0;
  }
  return v15;
}
