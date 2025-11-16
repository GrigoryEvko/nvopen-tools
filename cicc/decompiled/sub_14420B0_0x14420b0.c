// Function: sub_14420B0
// Address: 0x14420b0
//
__int64 __fastcall sub_14420B0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v4; // rsi
  int v5; // edx
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 j; // r13
  char v11; // dl
  unsigned __int64 i; // [rsp+10h] [rbp-60h]
  unsigned __int8 v14; // [rsp+1Fh] [rbp-51h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  char v18; // [rsp+38h] [rbp-38h]

  if ( a2 )
  {
    v14 = sub_1441AE0((_QWORD *)a1);
    if ( v14 )
    {
      v4 = sub_15E44B0(a2);
      if ( !v5 || sub_1441D60(a1, v4) )
      {
        v16 = a2 + 72;
        if ( (unsigned __int8)sub_1441AE0((_QWORD *)a1) )
        {
          v8 = *(_QWORD *)(a2 + 80);
          v6 = v8;
          if ( **(_DWORD **)(a1 + 8) == 1 )
          {
            for ( i = 0; v8 != v16; v8 = *(_QWORD *)(v8 + 8) )
            {
              if ( !v8 )
                BUG();
              v9 = *(_QWORD *)(v8 + 24);
              for ( j = v8 + 16; j != v9; i += v17 )
              {
                while ( 1 )
                {
                  if ( !v9 )
                    BUG();
                  v11 = *(_BYTE *)(v9 - 8);
                  if ( v11 == 78 || v11 == 29 )
                  {
                    sub_1441B50((__int64)&v17, a1, v9 - 24, 0);
                    if ( v18 )
                      break;
                  }
                  v9 = *(_QWORD *)(v9 + 8);
                  if ( j == v9 )
                    goto LABEL_27;
                }
                v9 = *(_QWORD *)(v9 + 8);
              }
LABEL_27:
              ;
            }
            if ( !sub_1441D60(a1, i) )
              return 0;
            v6 = *(_QWORD *)(a2 + 80);
          }
        }
        else
        {
          v6 = *(_QWORD *)(a2 + 80);
        }
        if ( v6 == v16 )
          return v14;
        while ( 1 )
        {
          v7 = v6 - 24;
          if ( !v6 )
            v7 = 0;
          if ( !sub_1442060(a1, v7, a3) )
            break;
          v6 = *(_QWORD *)(v6 + 8);
          if ( v6 == v16 )
            return v14;
        }
      }
    }
  }
  return 0;
}
