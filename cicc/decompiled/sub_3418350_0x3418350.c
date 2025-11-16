// Function: sub_3418350
// Address: 0x3418350
//
__int64 *__fastcall sub_3418350(__int64 *a1, unsigned __int64 a2, char a3)
{
  void *v3; // r9
  char *v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v8; // rax
  _BYTE *v9; // r10
  __int64 v10; // rax
  unsigned __int64 v11[3]; // [rsp+8h] [rbp-48h] BYREF
  char v12; // [rsp+24h] [rbp-2Ch] BYREF
  _BYTE v13[43]; // [rsp+25h] [rbp-2Bh] BYREF

  v3 = a1 + 2;
  if ( a2 )
  {
    v4 = v13;
    do
    {
      *--v4 = a2 % 0xA + 48;
      v8 = a2;
      a2 /= 0xAu;
    }
    while ( v8 > 9 );
    if ( !a3 )
    {
LABEL_10:
      v9 = (_BYTE *)(v13 - v4);
      *a1 = (__int64)v3;
      v11[0] = v13 - v4;
      if ( (unsigned __int64)(v13 - v4) <= 0xF )
      {
        if ( v9 == (_BYTE *)1 )
          goto LABEL_4;
        if ( !v9 )
          goto LABEL_5;
      }
      else
      {
        v10 = sub_22409D0((__int64)a1, v11, 0);
        *a1 = v10;
        v3 = (void *)v10;
        a1[2] = v11[0];
      }
      memcpy(v3, v4, v13 - v4);
      goto LABEL_5;
    }
LABEL_9:
    *--v4 = 45;
    goto LABEL_10;
  }
  v12 = 48;
  if ( a3 )
  {
    v4 = &v12;
    goto LABEL_9;
  }
  *a1 = (__int64)v3;
  v4 = &v12;
  v11[0] = 1;
LABEL_4:
  *((_BYTE *)a1 + 16) = *v4;
LABEL_5:
  v5 = v11[0];
  v6 = *a1;
  a1[1] = v11[0];
  *(_BYTE *)(v6 + v5) = 0;
  return a1;
}
