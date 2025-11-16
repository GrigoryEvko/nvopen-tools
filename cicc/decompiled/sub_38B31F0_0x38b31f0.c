// Function: sub_38B31F0
// Address: 0x38b31f0
//
__int64 __fastcall sub_38B31F0(__int64 a1, __int64 a2)
{
  int v3; // eax
  const char *v4; // rax
  unsigned __int64 v5; // rsi
  int v6; // eax
  int v7; // eax
  const char *v8; // [rsp+0h] [rbp-40h] BYREF
  char v9; // [rsp+10h] [rbp-30h]
  char v10; // [rsp+11h] [rbp-2Fh]

  if ( (unsigned __int8)sub_388AF10(a1, 355, "expected 'wpdRes' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388AF10(a1, 343, "expected 'kind' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
  {
    return 1;
  }
  v3 = *(_DWORD *)(a1 + 64);
  switch ( v3 )
  {
    case 357:
      *(_DWORD *)a2 = 1;
      break;
    case 358:
      *(_DWORD *)a2 = 2;
      break;
    case 356:
      *(_DWORD *)a2 = 0;
      break;
    default:
      v10 = 1;
      v4 = "unexpected WholeProgramDevirtResolution kind";
      goto LABEL_11;
  }
  v6 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v6;
  if ( v6 != 4 )
    return sub_388AF10(a1, 13, "expected ')' here");
  while ( 1 )
  {
    v7 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v7;
    if ( v7 != 359 )
      break;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
      || (unsigned __int8)sub_388B0A0(a1, (unsigned __int64 *)(a2 + 8)) )
    {
      return 1;
    }
LABEL_16:
    if ( *(_DWORD *)(a1 + 64) != 4 )
      return sub_388AF10(a1, 13, "expected ')' here");
  }
  if ( v7 == 360 )
  {
    if ( (unsigned __int8)sub_38B2A50(a1, (_QWORD *)(a2 + 40)) )
      return 1;
    goto LABEL_16;
  }
  v10 = 1;
  v4 = "expected optional WholeProgramDevirtResolution field";
LABEL_11:
  v5 = *(_QWORD *)(a1 + 56);
  v8 = v4;
  v9 = 3;
  return sub_38814C0(a1 + 8, v5, (__int64)&v8);
}
