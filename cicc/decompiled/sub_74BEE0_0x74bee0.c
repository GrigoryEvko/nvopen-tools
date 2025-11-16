// Function: sub_74BEE0
// Address: 0x74bee0
//
__int64 __fastcall sub_74BEE0(__int64 a1, void (__fastcall **a2)(const char *))
{
  char v2; // al
  __int64 *v3; // r13
  __int64 v5; // r13

  v2 = *(_BYTE *)(a1 + 173);
  if ( v2 == 6 )
  {
    v5 = *(_QWORD *)(a1 + 184);
    (*a2)("__uuidof(");
    if ( v5 )
    {
      sub_74B930(v5, (__int64)a2);
      return ((__int64 (__fastcall *)(char *, void (__fastcall **)(const char *)))*a2)(")", a2);
    }
    goto LABEL_6;
  }
  if ( v2 != 12 )
    sub_721090();
  v3 = sub_72F1F0(a1);
  ((void (__fastcall *)(const char *, void (__fastcall **)(const char *)))*a2)("__uuidof(", a2);
  if ( !v3 )
  {
LABEL_6:
    ((void (__fastcall *)(char *, void (__fastcall **)(const char *)))*a2)("0", a2);
    return ((__int64 (__fastcall *)(char *, void (__fastcall **)(const char *)))*a2)(")", a2);
  }
  sub_747C50((__int64)v3, (__int64)a2);
  return ((__int64 (__fastcall *)(char *, void (__fastcall **)(_QWORD)))*a2)(")", (void (__fastcall **)(_QWORD))a2);
}
