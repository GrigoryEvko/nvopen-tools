// Function: sub_12B9B00
// Address: 0x12b9b00
//
__int64 __fastcall sub_12B9B00(__int64 *a1, char a2, _BYTE *a3, _BYTE *a4)
{
  char *v7; // rcx
  _BYTE *v8; // rdx
  char *v9; // rsi
  char v10; // al
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v13[0] = a4 - a3;
  if ( (unsigned __int64)(a4 - a3) > 0xF )
  {
    v12 = sub_22409D0(a1, v13, 0);
    *a1 = v12;
    v7 = (char *)v12;
    a1[2] = v13[0];
  }
  else
  {
    v7 = (char *)*a1;
  }
  if ( a4 != a3 )
  {
    v8 = a3;
    v9 = &v7[a4 - a3];
    while ( 1 )
    {
      v10 = *v8 ^ (-109 * (((_BYTE)v8 - a2 + 97) ^ 0xC5));
      if ( (unsigned __int8)(v10 - 97) <= 0xCu )
        goto LABEL_11;
      if ( v10 <= 64 )
        break;
      if ( v10 <= 77 )
      {
LABEL_11:
        ++v7;
        ++v8;
        *(v7 - 1) = v10 + 13;
        if ( v7 == v9 )
        {
LABEL_12:
          v7 = (char *)*a1;
          goto LABEL_13;
        }
      }
      else
      {
        if ( (unsigned __int8)(v10 - 110) > 0xCu )
          break;
LABEL_8:
        v10 -= 13;
LABEL_9:
        *v7++ = v10;
        ++v8;
        if ( v7 == v9 )
          goto LABEL_12;
      }
    }
    if ( (unsigned __int8)(v10 - 78) > 0xCu )
      goto LABEL_9;
    goto LABEL_8;
  }
LABEL_13:
  result = v13[0];
  a1[1] = v13[0];
  v7[result] = 0;
  return result;
}
