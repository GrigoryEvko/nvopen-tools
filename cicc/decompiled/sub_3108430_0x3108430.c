// Function: sub_3108430
// Address: 0x3108430
//
bool __fastcall sub_3108430(__int64 a1, __int64 a2)
{
  char v2; // al
  bool result; // al
  __int64 v4; // rax
  __int64 v5; // [rsp-58h] [rbp-58h] BYREF
  __int64 v6; // [rsp-50h] [rbp-50h]
  __int64 v7; // [rsp-48h] [rbp-48h]
  __int64 v8; // [rsp-40h] [rbp-40h]
  __int64 v9; // [rsp-38h] [rbp-38h]
  __int64 v10; // [rsp-30h] [rbp-30h]

  v2 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x15u || v2 == 60 )
    return 0;
  if ( v2 == 22
    && ((unsigned __int8)sub_B2BAE0(a1) || (unsigned __int8)sub_B2D6E0(a1) || (unsigned __int8)sub_B2D720(a1)) )
  {
    return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 14 )
    return 0;
  v5 = a1;
  v6 = -1;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  if ( !(unsigned __int8)sub_CF5020(a2, (__int64)&v5, 0) )
    return 0;
  result = 1;
  if ( *(_BYTE *)a1 == 61 )
  {
    v4 = *(_QWORD *)(a1 - 32);
    v6 = -1;
    v5 = v4;
    v7 = 0;
    v8 = 0;
    v9 = 0;
    v10 = 0;
    return (unsigned __int8)sub_CF5020(a2, (__int64)&v5, 0) != 0;
  }
  return result;
}
