// Function: sub_B16B10
// Address: 0xb16b10
//
unsigned __int64 __fastcall sub_B16B10(__int64 *a1, _BYTE *a2, __int64 a3, unsigned __int64 a4)
{
  void *v6; // rdi
  char *v7; // r12
  unsigned __int64 result; // rax
  size_t v9; // r14
  __int64 v10; // rax
  _BYTE *v11; // [rsp+8h] [rbp-48h] BYREF
  char v12; // [rsp+24h] [rbp-2Ch] BYREF
  _BYTE v13[43]; // [rsp+25h] [rbp-2Bh] BYREF

  *a1 = (__int64)(a1 + 2);
  sub_B14B30(a1, a2, (__int64)&a2[a3]);
  if ( a4 )
  {
    v7 = v13;
    do
    {
      *--v7 = a4 % 0xA + 48;
      result = a4;
      a4 /= 0xAu;
    }
    while ( result > 9 );
    v9 = v13 - v7;
    v6 = a1 + 6;
    a1[4] = (__int64)(a1 + 6);
    v11 = (_BYTE *)(v13 - v7);
    if ( (unsigned __int64)(v13 - v7) <= 0xF )
    {
      if ( v9 == 1 )
        goto LABEL_3;
      if ( !v9 )
        goto LABEL_4;
    }
    else
    {
      v10 = sub_22409D0(a1 + 4, &v11, 0);
      a1[4] = v10;
      v6 = (void *)v10;
      a1[6] = (__int64)v11;
    }
    result = (unsigned __int64)memcpy(v6, v7, v9);
    v9 = (size_t)v11;
    v6 = (void *)a1[4];
    goto LABEL_4;
  }
  v6 = a1 + 6;
  v12 = 48;
  v7 = &v12;
  a1[4] = (__int64)(a1 + 6);
LABEL_3:
  result = (unsigned __int8)*v7;
  v9 = 1;
  *((_BYTE *)a1 + 48) = result;
LABEL_4:
  a1[5] = v9;
  *((_BYTE *)v6 + v9) = 0;
  a1[8] = 0;
  a1[9] = 0;
  return result;
}
