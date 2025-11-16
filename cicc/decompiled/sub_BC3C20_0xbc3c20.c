// Function: sub_BC3C20
// Address: 0xbc3c20
//
__int64 __fastcall sub_BC3C20(unsigned int *a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  char v5; // cl
  unsigned int v6; // r14d
  char v7; // al
  __int64 v8; // rdx
  unsigned int v10; // r14d
  __int64 v11; // r13
  char v12; // al
  __int64 v13; // rsi
  char *v14; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  __int64 v17; // [rsp+18h] [rbp-28h]

  v14 = a3;
  v15 = a4;
  if ( !(unsigned __int8)sub_C92F10(&v14, "x", 1) )
  {
    if ( v15 )
    {
      v5 = *v14;
      if ( *v14 == 78 || v5 == 110 )
      {
        v6 = 1;
        ++v14;
        --v15;
LABEL_5:
        v7 = sub_C93B20(&v14, 10, &v16);
        v8 = 0;
        if ( !v7 )
          v8 = v16;
        return sub_C7F4A0(a2, *a1, v8, v6);
      }
      if ( v5 == 68 || v5 == 100 )
      {
        ++v14;
        --v15;
      }
    }
    v6 = 0;
    goto LABEL_5;
  }
  if ( v15 <= 1 )
  {
    if ( !v15 )
      goto LABEL_21;
    if ( *v14 != 120 )
    {
LABEL_14:
      if ( *v14 == 88 )
      {
        ++v14;
        --v15;
      }
      goto LABEL_21;
    }
    goto LABEL_34;
  }
  switch ( *(_WORD *)v14 )
  {
    case 0x2D78:
      v10 = 1;
      v14 += 2;
      v15 -= 2LL;
      goto LABEL_23;
    case 0x2D58:
      v10 = 0;
      v14 += 2;
      v15 -= 2LL;
      goto LABEL_23;
    case 0x2B78:
      v14 += 2;
      v15 -= 2LL;
LABEL_32:
      v10 = 3;
      goto LABEL_23;
  }
  if ( *v14 == 120 )
  {
LABEL_34:
    ++v14;
    --v15;
    goto LABEL_32;
  }
  if ( *(_WORD *)v14 != 11096 )
    goto LABEL_14;
  v14 += 2;
  v15 -= 2LL;
LABEL_21:
  v10 = 2;
LABEL_23:
  v11 = 0;
  if ( !(unsigned __int8)sub_C93B20(&v14, 10, &v16) )
    v11 = v16;
  v12 = sub_C7F6A0(v10);
  LOBYTE(v17) = 1;
  v13 = *a1;
  if ( v12 )
    v11 += 2;
  v16 = v11;
  return sub_C7F500(a2, v13, v10, v11, v17);
}
