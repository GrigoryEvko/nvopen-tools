// Function: sub_16E2C10
// Address: 0x16e2c10
//
__int64 *__fastcall sub_16E2C10(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r15d
  unsigned int v6; // r14d
  bool v7; // cf
  bool v8; // zf
  unsigned int v9; // [rsp+Ch] [rbp-54h]
  unsigned int v10; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v11; // [rsp+1Ch] [rbp-44h] BYREF
  unsigned int v12; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+24h] [rbp-3Ch] BYREF
  unsigned int v14; // [rsp+28h] [rbp-38h] BYREF
  _DWORD v15[13]; // [rsp+2Ch] [rbp-34h] BYREF

  if ( *(_DWORD *)(a2 + 40) != 1 )
    goto LABEL_2;
  sub_16E2390(a2, &v10, &v11, &v12);
  v5 = v10;
  v6 = v11;
  v9 = v12;
  sub_16E2390(a3, &v13, &v14, v15);
  v7 = v5 < v13;
  v8 = v5 == v13;
  if ( v5 == v13 )
  {
    v7 = v6 < v14;
    v8 = v6 == v14;
    if ( v6 == v14 )
    {
      if ( v9 == v15[0] )
        goto LABEL_2;
      v7 = v9 < v6;
      v8 = v9 == v6;
    }
  }
  if ( !v7 && !v8 )
  {
    *a1 = (__int64)(a1 + 2);
    sub_16DDF30(a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
    return a1;
  }
LABEL_2:
  *a1 = (__int64)(a1 + 2);
  sub_16DDF30(a1, *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  return a1;
}
