// Function: sub_13FD000
// Address: 0x13fd000
//
__int64 __fastcall sub_13FD000(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _BYTE *v7; // r14
  __int64 i; // r13
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // [rsp+8h] [rbp-68h]
  _BYTE *v13; // [rsp+10h] [rbp-60h] BYREF
  __int64 v14; // [rsp+18h] [rbp-58h]
  _BYTE v15[80]; // [rsp+20h] [rbp-50h] BYREF

  v14 = 0x400000000LL;
  v1 = *(_QWORD *)(a1 + 32);
  v13 = v15;
  v2 = *(_QWORD *)(*(_QWORD *)v1 + 8LL);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v3 = sub_1648700(v2);
    if ( (unsigned __int8)(*(_BYTE *)(v3 + 16) - 25) <= 9u )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
LABEL_6:
  v4 = *(_QWORD *)(v3 + 40);
  if ( sub_1377F70(a1 + 56, v4) )
  {
    v5 = (unsigned int)v14;
    if ( (unsigned int)v14 >= HIDWORD(v14) )
    {
      sub_16CD150(&v13, v15, 0, 8);
      v5 = (unsigned int)v14;
    }
    *(_QWORD *)&v13[8 * v5] = v4;
    LODWORD(v14) = v14 + 1;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v2 )
      goto LABEL_5;
  }
  else
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        break;
LABEL_5:
      v3 = sub_1648700(v2);
      if ( (unsigned __int8)(*(_BYTE *)(v3 + 16) - 25) <= 9u )
        goto LABEL_6;
    }
  }
  v6 = v13;
  v7 = &v13[8 * (unsigned int)v14];
  if ( v13 == v7 )
  {
LABEL_30:
    result = 0;
  }
  else
  {
    for ( i = 0; ; i = result )
    {
      v10 = sub_157EBA0(*v6);
      result = *(_QWORD *)(v10 + 48);
      if ( !result && *(__int16 *)(v10 + 18) >= 0 )
      {
        v7 = v13;
        goto LABEL_20;
      }
      result = sub_1625790(v10, 18);
      if ( !result || i && result != i )
      {
        v7 = v13;
        result = 0;
        goto LABEL_20;
      }
      if ( v7 == (_BYTE *)++v6 )
        break;
    }
    v11 = *(unsigned int *)(result + 8);
    v7 = v13;
    if ( !(_DWORD)v11 )
      goto LABEL_30;
    if ( *(_QWORD *)(result - 8 * v11) != result )
      result = 0;
  }
LABEL_20:
  if ( v7 != v15 )
  {
    v12 = result;
    _libc_free((unsigned __int64)v7);
    return v12;
  }
  return result;
}
