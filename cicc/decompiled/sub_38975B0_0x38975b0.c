// Function: sub_38975B0
// Address: 0x38975b0
//
__int64 __fastcall sub_38975B0(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int64 v2; // r12
  int v3; // eax
  unsigned __int64 v4; // rsi
  unsigned int v5; // r15d
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // [rsp+4h] [rbp-7Ch] BYREF
  __int64 v14; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int64 v15[4]; // [rsp+10h] [rbp-70h] BYREF
  unsigned int *v16[2]; // [rsp+30h] [rbp-50h] BYREF
  char v17; // [rsp+40h] [rbp-40h]
  char v18; // [rsp+41h] [rbp-3Fh]

  v1 = a1 + 8;
  v2 = *(_QWORD *)(a1 + 56);
  v3 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v3;
  if ( v3 != 370 )
  {
    v4 = *(_QWORD *)(a1 + 56);
    v18 = 1;
    v17 = 3;
    v16[0] = (unsigned int *)"expected attribute group id";
    return (unsigned int)sub_38814C0(v1, v4, (__int64)v16);
  }
  v7 = *(_DWORD *)(a1 + 104);
  memset(v15, 0, 24);
  v13 = v7;
  v14 = 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(v1);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' here")
    || (unsigned __int8)sub_388AF10(a1, 8, "expected '{' here") )
  {
    goto LABEL_5;
  }
  v8 = *(_QWORD *)(a1 + 1192);
  v9 = a1 + 1184;
  if ( !v8 )
  {
    v10 = a1 + 1184;
LABEL_17:
    v16[0] = &v13;
    v10 = sub_3896710((_QWORD *)(a1 + 1176), v10, v16);
    goto LABEL_18;
  }
  v10 = a1 + 1184;
  do
  {
    if ( *(_DWORD *)(v8 + 32) < v13 )
    {
      v8 = *(_QWORD *)(v8 + 24);
    }
    else
    {
      v10 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
  }
  while ( v8 );
  if ( v9 == v10 || v13 < *(_DWORD *)(v10 + 32) )
    goto LABEL_17;
LABEL_18:
  if ( !(unsigned __int8)sub_388FCA0(a1, (__m128i *)(v10 + 40), (__int64)v15, 1, &v14) )
  {
    v5 = sub_388AF10(a1, 9, "expected end of attribute group");
    if ( !(_BYTE)v5 )
    {
      v11 = *(_QWORD *)(a1 + 1192);
      if ( v11 )
      {
        v12 = a1 + 1184;
        do
        {
          if ( *(_DWORD *)(v11 + 32) < v13 )
          {
            v11 = *(_QWORD *)(v11 + 24);
          }
          else
          {
            v12 = v11;
            v11 = *(_QWORD *)(v11 + 16);
          }
        }
        while ( v11 );
        if ( v12 != v9 && v13 >= *(_DWORD *)(v12 + 32) )
          goto LABEL_29;
      }
      else
      {
        v12 = a1 + 1184;
      }
      v16[0] = &v13;
      v12 = sub_3896710((_QWORD *)(a1 + 1176), v12, v16);
LABEL_29:
      if ( !sub_1560CB0((_QWORD *)(v12 + 40)) )
      {
        v18 = 1;
        v17 = 3;
        v16[0] = (unsigned int *)"attribute group has no attributes";
        v5 = sub_38814C0(v1, v2, (__int64)v16);
      }
      goto LABEL_6;
    }
  }
LABEL_5:
  v5 = 1;
LABEL_6:
  if ( v15[0] )
    j_j___libc_free_0(v15[0]);
  return v5;
}
