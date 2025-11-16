// Function: sub_ED8FE0
// Address: 0xed8fe0
//
__int64 __fastcall sub_ED8FE0(unsigned __int64 *a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  int v5; // r14d
  char v6; // cl
  char v7; // al
  unsigned __int64 v8; // rdx
  int v10; // r14d
  unsigned __int64 v11; // r13
  bool v12; // al
  unsigned __int64 v13; // rsi
  char *v14; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v15; // [rsp+8h] [rbp-38h]
  unsigned __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  __int64 v17; // [rsp+18h] [rbp-28h]

  v14 = a3;
  v15 = a4;
  if ( !(unsigned __int8)sub_C92F10(&v14, (__int64)"x", 1u) )
  {
    if ( !v15 )
      goto LABEL_3;
    v6 = *v14;
    if ( *v14 == 78 || v6 == 110 )
    {
      v5 = 1;
      ++v14;
      --v15;
      goto LABEL_7;
    }
    if ( v6 == 68 )
    {
      ++v14;
      --v15;
    }
    else
    {
LABEL_3:
      sub_95CB50((const void **)&v14, "d", 1u);
    }
    v5 = 0;
LABEL_7:
    v7 = sub_C93B20((__int64 *)&v14, 0xAu, &v16);
    v8 = 0;
    if ( !v7 )
      v8 = v16;
    return sub_C7F4D0(a2, *a1, v8, v5);
  }
  if ( v15 <= 1 )
    goto LABEL_23;
  if ( *(_WORD *)v14 == 11640 )
  {
    v10 = 1;
    v14 += 2;
    v15 -= 2LL;
    goto LABEL_14;
  }
  if ( *(_WORD *)v14 != 11608 )
  {
    if ( *(_WORD *)v14 == 11128 )
    {
      v14 += 2;
      v15 -= 2LL;
LABEL_24:
      v10 = 3;
      goto LABEL_14;
    }
LABEL_23:
    if ( !(unsigned __int8)sub_95CB50((const void **)&v14, "x", 1u) )
    {
      if ( !(unsigned __int8)sub_95CB50((const void **)&v14, "X+", 2u) )
        sub_95CB50((const void **)&v14, "X", 1u);
      v10 = 2;
      goto LABEL_14;
    }
    goto LABEL_24;
  }
  v10 = 0;
  v14 += 2;
  v15 -= 2LL;
LABEL_14:
  v11 = 0;
  if ( !(unsigned __int8)sub_C93B20((__int64 *)&v14, 0xAu, &v16) )
    v11 = v16;
  v12 = sub_C7F6A0(v10);
  LOBYTE(v17) = 1;
  v13 = *a1;
  if ( v12 )
    v11 += 2LL;
  v16 = v11;
  return sub_C7F500(a2, v13, v10, v11, 1);
}
