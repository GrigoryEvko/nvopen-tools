// Function: sub_2EB5800
// Address: 0x2eb5800
//
__int64 *__fastcall sub_2EB5800(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r15d
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // r15d
  int v15; // r15d
  __int64 v16; // rcx
  __int64 v17; // r8
  int v18; // r15d
  __int64 *result; // rax
  char v20; // r8
  char v21; // r8
  bool v22; // zf
  _BYTE *v24; // [rsp+10h] [rbp-80h] BYREF
  int v25; // [rsp+18h] [rbp-78h]
  _BYTE v26[112]; // [rsp+20h] [rbp-70h] BYREF

  v6 = a1;
  v7 = (char *)a2 - (char *)a1;
  v8 = v7 >> 3;
  if ( v7 >> 5 <= 0 )
  {
LABEL_19:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
          return a2;
LABEL_27:
        v22 = (unsigned __int8)sub_2EB57C0(*v6, a3, a3, a4) == 0;
        result = a2;
        if ( !v22 )
          return v6;
        return result;
      }
      v20 = sub_2EB57C0(*v6, a3, a3, a4);
      result = v6;
      if ( v20 )
        return result;
      ++v6;
    }
    v21 = sub_2EB57C0(*v6, a3, a3, a4);
    result = v6;
    if ( v21 )
      return result;
    ++v6;
    goto LABEL_27;
  }
  while ( 1 )
  {
    sub_2EB5530(&v24, *v6, a3, a4, a5);
    v18 = v25;
    if ( v24 != v26 )
      _libc_free((unsigned __int64)v24);
    if ( v18 )
      return v6;
    sub_2EB5530(&v24, v6[1], a3, v16, v17);
    v11 = v25;
    if ( v24 != v26 )
      _libc_free((unsigned __int64)v24);
    if ( v11 )
      return v6 + 1;
    sub_2EB5530(&v24, v6[2], a3, v9, v10);
    v14 = v25;
    if ( v24 != v26 )
      _libc_free((unsigned __int64)v24);
    if ( v14 )
      return v6 + 2;
    sub_2EB5530(&v24, v6[3], a3, v12, v13);
    v15 = v25;
    if ( v24 != v26 )
      _libc_free((unsigned __int64)v24);
    if ( v15 )
      return v6 + 3;
    v6 += 4;
    if ( &a1[4 * (v7 >> 5)] == v6 )
    {
      v8 = a2 - v6;
      goto LABEL_19;
    }
  }
}
