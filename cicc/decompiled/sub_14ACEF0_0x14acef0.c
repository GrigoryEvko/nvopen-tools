// Function: sub_14ACEF0
// Address: 0x14acef0
//
__int64 __fastcall sub_14ACEF0(__int64 a1, _QWORD *a2, __int64 a3, unsigned int a4)
{
  unsigned int v5; // r12d
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  const void *v10; // r15
  size_t v11; // r14
  size_t v12; // rdx
  _BYTE *v13; // rax
  unsigned __int64 v14; // rax
  bool v16; // sf
  _QWORD v17[10]; // [rsp+0h] [rbp-50h] BYREF

  v5 = sub_14ACAF0(a1, (__int64)v17, 8u, a3);
  if ( !(_BYTE)v5 )
    return v5;
  if ( !v17[0] )
  {
    if ( (_BYTE)a4 )
    {
      *a2 = 0;
      v5 = a4;
      a2[1] = 0;
    }
    else if ( v17[2] == 1 )
    {
      a2[1] = 1;
      *a2 = byte_3F871B3;
    }
    else
    {
      return 0;
    }
    return v5;
  }
  v6 = sub_1595920();
  v8 = v17[1];
  a2[1] = v7;
  if ( v8 > v7 )
  {
    a2[1] = 0;
    *a2 = v7 + v6;
    if ( !(_BYTE)a4 )
      return v5;
    goto LABEL_17;
  }
  v9 = v7 - v8;
  v10 = (const void *)(v6 + v8);
  *a2 = v10;
  v11 = v9;
  if ( v9 != -1 )
  {
    a2[1] = v9;
    if ( !(_BYTE)a4 )
      return v5;
    v16 = v9 < 0;
    if ( v9 )
    {
      v12 = 0x7FFFFFFFFFFFFFFFLL;
      if ( !v16 )
        v12 = v11;
      goto LABEL_7;
    }
LABEL_17:
    v11 = 0;
LABEL_10:
    a2[1] = v11;
    return v5;
  }
  a2[1] = -1;
  if ( (_BYTE)a4 )
  {
    v12 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_7:
    v13 = memchr(v10, 0, v12);
    if ( v13 )
    {
      v14 = v13 - (_BYTE *)v10;
      if ( v11 > v14 )
        v11 = v14;
    }
    goto LABEL_10;
  }
  return v5;
}
