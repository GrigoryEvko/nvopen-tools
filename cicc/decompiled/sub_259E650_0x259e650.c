// Function: sub_259E650
// Address: 0x259e650
//
char __fastcall sub_259E650(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char result; // al
  __int64 v5; // rdx
  char v6; // [rsp-3Ah] [rbp-3Ah] BYREF
  bool v7; // [rsp-39h] [rbp-39h] BYREF
  __m128i v8[3]; // [rsp-38h] [rbp-38h] BYREF

  if ( !a3 )
    return 1;
  result = sub_F509B0((unsigned __int8 *)a3, 0);
  if ( result )
    return 1;
  if ( *(_BYTE *)a3 != 85 )
  {
    result = *(_BYTE *)a3 == 34 || *(_BYTE *)a3 == 40;
    if ( !result )
      return result;
    goto LABEL_5;
  }
  v5 = *(_QWORD *)(a3 - 32);
  if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a3 + 80) || (*(_BYTE *)(v5 + 33) & 0x20) == 0 )
  {
LABEL_5:
    sub_250D230((unsigned __int64 *)v8, a3, 5, 0);
    result = sub_259E0A0(a2, a1, v8, 1, &v6, 0, 0);
    if ( result )
      return sub_252A800(a2, v8, a1, &v7);
  }
  return result;
}
