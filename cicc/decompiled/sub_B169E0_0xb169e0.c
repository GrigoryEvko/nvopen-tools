// Function: sub_B169E0
// Address: 0xb169e0
//
unsigned __int64 __fastcall sub_B169E0(__int64 *a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // rcx
  void *v7; // rdi
  char *v8; // r13
  unsigned __int64 result; // rax
  size_t v10; // r14
  __int64 v11; // rax
  _BYTE *v12; // [rsp+8h] [rbp-48h] BYREF
  char v13; // [rsp+24h] [rbp-2Ch] BYREF
  _BYTE v14[43]; // [rsp+25h] [rbp-2Bh] BYREF

  *a1 = (__int64)(a1 + 2);
  sub_B14B30(a1, a2, (__int64)&a2[a3]);
  v6 = a4;
  if ( a4 )
  {
    v8 = v14;
    do
    {
      *--v8 = v6 % 0xA + 48;
      result = v6;
      v6 /= 0xAu;
    }
    while ( result > 9 );
    v10 = v14 - v8;
    v7 = a1 + 6;
    a1[4] = (__int64)(a1 + 6);
    v12 = (_BYTE *)(v14 - v8);
    if ( (unsigned __int64)(v14 - v8) <= 0xF )
    {
      if ( v10 == 1 )
        goto LABEL_3;
      if ( !v10 )
        goto LABEL_4;
    }
    else
    {
      v11 = sub_22409D0(a1 + 4, &v12, 0);
      a1[4] = v11;
      v7 = (void *)v11;
      a1[6] = (__int64)v12;
    }
    result = (unsigned __int64)memcpy(v7, v8, v10);
    v10 = (size_t)v12;
    v7 = (void *)a1[4];
    goto LABEL_4;
  }
  v7 = a1 + 6;
  v13 = 48;
  v8 = &v13;
  a1[4] = (__int64)(a1 + 6);
LABEL_3:
  result = (unsigned __int8)*v8;
  v10 = 1;
  *((_BYTE *)a1 + 48) = result;
LABEL_4:
  a1[5] = v10;
  *((_BYTE *)v7 + v10) = 0;
  a1[8] = 0;
  a1[9] = 0;
  return result;
}
