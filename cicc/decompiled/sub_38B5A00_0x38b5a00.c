// Function: sub_38B5A00
// Address: 0x38b5a00
//
__int64 __fastcall sub_38B5A00(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  int v4; // eax
  unsigned __int64 v5; // rsi
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+10h] [rbp-30h]
  char v8; // [rsp+11h] [rbp-2Fh]

  v2 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' in typeIdInfo") )
  {
    return 1;
  }
  v4 = *(_DWORD *)(a1 + 64);
  while ( 2 )
  {
    switch ( v4 )
    {
      case 332:
        if ( (unsigned __int8)sub_38B4340(a1, a2) )
          return 1;
        goto LABEL_7;
      case 333:
        if ( (unsigned __int8)sub_38B5120(a1, 333, a2 + 3) )
          return 1;
        goto LABEL_7;
      case 334:
        if ( (unsigned __int8)sub_38B5120(a1, 334, a2 + 6) )
          return 1;
        goto LABEL_7;
      case 335:
        if ( (unsigned __int8)sub_38B5500(a1, 335, (__int64)(a2 + 9)) )
          return 1;
        goto LABEL_7;
      case 336:
        if ( (unsigned __int8)sub_38B5500(a1, 336, (__int64)(a2 + 12)) )
          return 1;
LABEL_7:
        if ( *(_DWORD *)(a1 + 64) == 4 )
        {
          v4 = sub_3887100(v2);
          *(_DWORD *)(a1 + 64) = v4;
          continue;
        }
        return sub_388AF10(a1, 13, "expected ')' in typeIdInfo");
      default:
        v5 = *(_QWORD *)(a1 + 56);
        v8 = 1;
        v7 = 3;
        v6 = "invalid typeIdInfo list type";
        return sub_38814C0(v2, v5, (__int64)&v6);
    }
  }
}
