// Function: sub_38936D0
// Address: 0x38936d0
//
__int64 __fastcall sub_38936D0(__int64 a1)
{
  unsigned __int64 v1; // r13
  unsigned int v2; // r15d
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // [rsp+4h] [rbp-5Ch] BYREF
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  unsigned int *v14[2]; // [rsp+10h] [rbp-50h] BYREF
  char v15; // [rsp+20h] [rbp-40h]
  char v16; // [rsp+21h] [rbp-3Fh]

  v1 = *(_QWORD *)(a1 + 56);
  v12 = *(_DWORD *)(a1 + 104);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after name")
    || (unsigned __int8)sub_388AF10(a1, 199, "expected 'type' after '='") )
  {
    return 1;
  }
  v4 = *(_QWORD *)(a1 + 776);
  v13 = 0;
  v5 = a1 + 768;
  if ( !v4 )
  {
    v6 = a1 + 768;
LABEL_12:
    v14[0] = &v12;
    v6 = sub_3893610((_QWORD *)(a1 + 760), v6, v14);
    goto LABEL_13;
  }
  v6 = a1 + 768;
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v4 + 16);
      v8 = *(_QWORD *)(v4 + 24);
      if ( *(_DWORD *)(v4 + 32) >= v12 )
        break;
      v4 = *(_QWORD *)(v4 + 24);
      if ( !v8 )
        goto LABEL_10;
    }
    v6 = v4;
    v4 = *(_QWORD *)(v4 + 16);
  }
  while ( v7 );
LABEL_10:
  if ( v6 == v5 || v12 < *(_DWORD *)(v6 + 32) )
    goto LABEL_12;
LABEL_13:
  v2 = sub_38925C0(a1, v1, byte_3F871B3, 0, (__int64 *)(v6 + 40), &v13);
  if ( (_BYTE)v2 )
    return 1;
  if ( *(_BYTE *)(v13 + 8) == 13 )
    return v2;
  v9 = *(_QWORD *)(a1 + 776);
  if ( v9 )
  {
    v10 = a1 + 768;
    do
    {
      if ( *(_DWORD *)(v9 + 32) < v12 )
      {
        v9 = *(_QWORD *)(v9 + 24);
      }
      else
      {
        v10 = v9;
        v9 = *(_QWORD *)(v9 + 16);
      }
    }
    while ( v9 );
    if ( v5 != v10 && v12 >= *(_DWORD *)(v10 + 32) )
      goto LABEL_24;
  }
  else
  {
    v10 = a1 + 768;
  }
  v14[0] = &v12;
  v10 = sub_3893610((_QWORD *)(a1 + 760), v10, v14);
LABEL_24:
  if ( *(_QWORD *)(v10 + 40) )
  {
    v16 = 1;
    v15 = 3;
    v14[0] = (unsigned int *)"non-struct types may not be recursive";
    return (unsigned int)sub_38814C0(a1 + 8, v1, (__int64)v14);
  }
  else
  {
    v11 = v13;
    *(_QWORD *)(v10 + 48) = 0;
    *(_QWORD *)(v10 + 40) = v11;
  }
  return v2;
}
