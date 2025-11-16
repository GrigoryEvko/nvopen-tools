// Function: sub_EAD580
// Address: 0xead580
//
__int64 __fastcall sub_EAD580(__int64 a1, __int64 a2)
{
  __int64 v3; // [rsp+10h] [rbp-50h] BYREF
  __int64 v4; // [rsp+18h] [rbp-48h] BYREF
  const char *v5; // [rsp+20h] [rbp-40h] BYREF
  char v6; // [rsp+40h] [rbp-20h]
  char v7; // [rsp+41h] [rbp-1Fh]

  v3 = 0;
  v4 = 0;
  if ( (unsigned __int8)sub_EAD290(a1, &v3, a2) )
    return 1;
  v7 = 1;
  v6 = 3;
  v5 = "expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, &v5)
    || (unsigned __int8)sub_EAC8B0(a1, &v4)
    || (unsigned __int8)sub_ECE000(a1) )
  {
    return 1;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 232) + 896LL))(
    *(_QWORD *)(a1 + 232),
    v3,
    v4,
    a2);
  return 0;
}
