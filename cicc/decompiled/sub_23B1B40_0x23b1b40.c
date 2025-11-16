// Function: sub_23B1B40
// Address: 0x23b1b40
//
__int64 __fastcall sub_23B1B40(__int64 a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  signed int *v4; // rbx
  int v5; // r14d
  char v6; // al
  unsigned __int64 v7; // rdx
  int v9; // r14d
  unsigned __int64 v10; // r12
  bool v11; // al
  char *v12; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v13; // [rsp+8h] [rbp-38h]
  unsigned __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  __int64 v15; // [rsp+18h] [rbp-28h]

  v12 = a3;
  v4 = *(signed int **)(a1 + 8);
  v13 = a4;
  if ( !(unsigned __int8)sub_C92F10(&v12, (__int64)"x", 1u) )
  {
    if ( v13 && *v12 == 78 )
    {
      ++v12;
      --v13;
    }
    else if ( !(unsigned __int8)sub_95CB50((const void **)&v12, "n", 1u) )
    {
      if ( !(unsigned __int8)sub_95CB50((const void **)&v12, "D", 1u) )
        sub_95CB50((const void **)&v12, "d", 1u);
      v5 = 0;
      goto LABEL_5;
    }
    v5 = 1;
LABEL_5:
    v6 = sub_C93B20((__int64 *)&v12, 0xAu, &v14);
    v7 = 0;
    if ( !v6 )
      v7 = v14;
    return sub_C7F4B0(a2, *v4, v7, v5);
  }
  v9 = 1;
  if ( !(unsigned __int8)sub_95CB50((const void **)&v12, "x-", 2u) )
  {
    v9 = 0;
    if ( !(unsigned __int8)sub_95CB50((const void **)&v12, "X-", 2u) )
    {
      if ( (unsigned __int8)sub_95CB50((const void **)&v12, "x+", 2u)
        || (unsigned __int8)sub_95CB50((const void **)&v12, "x", 1u) )
      {
        v9 = 3;
      }
      else
      {
        if ( v13 > 1 && *(_WORD *)v12 == 11096 )
        {
          v12 += 2;
          v13 -= 2LL;
        }
        else
        {
          sub_95CB50((const void **)&v12, "X", 1u);
        }
        v9 = 2;
      }
    }
  }
  v10 = 0;
  if ( !(unsigned __int8)sub_C93B20((__int64 *)&v12, 0xAu, &v14) )
    v10 = v14;
  v11 = sub_C7F6A0(v9);
  LOBYTE(v15) = 1;
  if ( v11 )
    v10 += 2LL;
  v14 = v10;
  return sub_C7F500(a2, *v4, v9, v10, 1);
}
