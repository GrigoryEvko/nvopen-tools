// Function: sub_125C410
// Address: 0x125c410
//
__int64 *__fastcall sub_125C410(__int64 *a1, _BYTE *a2, unsigned __int64 a3)
{
  char *v4; // rdx
  _BYTE *v6; // rcx
  char *v7; // rdi
  char v8; // al
  unsigned __int64 v9; // rax
  __int64 v11; // rax
  unsigned __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = (char *)(a1 + 2);
  *a1 = (__int64)(a1 + 2);
  v12[0] = a3;
  if ( a3 > 0xF )
  {
    v11 = sub_22409D0(a1, v12, 0);
    *a1 = v11;
    v4 = (char *)v11;
    a1[2] = v12[0];
  }
  if ( a3 )
  {
    v6 = a2;
    v7 = &v4[a3];
    while ( 1 )
    {
      v8 = *v6 ^ (-109 * (((_BYTE)v6 - (_BYTE)a2 + 97) ^ 0xC5));
      if ( (unsigned __int8)(v8 - 97) <= 0xCu )
        goto LABEL_11;
      if ( v8 <= 64 )
        break;
      if ( v8 <= 77 )
      {
LABEL_11:
        ++v4;
        ++v6;
        *(v4 - 1) = v8 + 13;
        if ( v4 == v7 )
        {
LABEL_12:
          v4 = (char *)*a1;
          goto LABEL_13;
        }
      }
      else
      {
        if ( (unsigned __int8)(v8 - 110) > 0xCu )
          break;
LABEL_8:
        v8 -= 13;
LABEL_9:
        *v4++ = v8;
        ++v6;
        if ( v4 == v7 )
          goto LABEL_12;
      }
    }
    if ( (unsigned __int8)(v8 - 78) > 0xCu )
      goto LABEL_9;
    goto LABEL_8;
  }
LABEL_13:
  v9 = v12[0];
  a1[1] = v12[0];
  v4[v9] = 0;
  return a1;
}
