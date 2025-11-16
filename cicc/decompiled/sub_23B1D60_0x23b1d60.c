// Function: sub_23B1D60
// Address: 0x23b1d60
//
__int64 __fastcall sub_23B1D60(__int64 a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  int v5; // r14d
  char v6; // al
  unsigned __int64 v7; // rdx
  int v9; // r14d
  unsigned __int64 v10; // r12
  bool v11; // al
  unsigned __int64 v12; // rsi
  char *v13; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v14; // [rsp+8h] [rbp-38h]
  unsigned __int64 v15; // [rsp+10h] [rbp-30h] BYREF
  __int64 v16; // [rsp+18h] [rbp-28h]

  v13 = a3;
  v14 = a4;
  if ( !(unsigned __int8)sub_C92F10(&v13, (__int64)"x", 1u) )
  {
    if ( v14 && *v13 == 78 )
    {
      ++v13;
      --v14;
    }
    else if ( !(unsigned __int8)sub_95CB50((const void **)&v13, "n", 1u) )
    {
      if ( !(unsigned __int8)sub_95CB50((const void **)&v13, "D", 1u) )
        sub_95CB50((const void **)&v13, "d", 1u);
      v5 = 0;
      goto LABEL_5;
    }
    v5 = 1;
LABEL_5:
    v6 = sub_C93B20((__int64 *)&v13, 0xAu, &v15);
    v7 = 0;
    if ( !v6 )
      v7 = v15;
    return sub_C7F4E0(a2, *(_QWORD *)(a1 + 8), v7, v5);
  }
  v9 = 1;
  if ( !(unsigned __int8)sub_95CB50((const void **)&v13, "x-", 2u) )
  {
    v9 = 0;
    if ( !(unsigned __int8)sub_95CB50((const void **)&v13, "X-", 2u) )
    {
      if ( (unsigned __int8)sub_95CB50((const void **)&v13, "x+", 2u)
        || (unsigned __int8)sub_95CB50((const void **)&v13, "x", 1u) )
      {
        v9 = 3;
      }
      else
      {
        if ( v14 > 1 && *(_WORD *)v13 == 11096 )
        {
          v13 += 2;
          v14 -= 2LL;
        }
        else
        {
          sub_95CB50((const void **)&v13, "X", 1u);
        }
        v9 = 2;
      }
    }
  }
  v10 = 0;
  if ( !(unsigned __int8)sub_C93B20((__int64 *)&v13, 0xAu, &v15) )
    v10 = v15;
  v11 = sub_C7F6A0(v9);
  LOBYTE(v16) = 1;
  v12 = *(_QWORD *)(a1 + 8);
  if ( v11 )
    v10 += 2LL;
  v15 = v10;
  return sub_C7F500(a2, v12, v9, v10, 1);
}
