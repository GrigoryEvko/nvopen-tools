// Function: sub_74BF80
// Address: 0x74bf80
//
__int64 __fastcall sub_74BF80(__int64 a1, void (__fastcall **a2)(const char *))
{
  char v2; // al
  __int64 *v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13

  v2 = *(_BYTE *)(a1 + 173);
  if ( v2 == 6 )
  {
    v4 = *(_QWORD *)(a1 + 184);
    (*a2)("typeid(");
    goto LABEL_7;
  }
  if ( v2 != 12 )
    goto LABEL_9;
  v3 = sub_72F1F0(a1);
  v4 = *(_QWORD *)(a1 + 184);
  v5 = (__int64)v3;
  ((void (__fastcall *)(const char *, void (__fastcall **)(const char *)))*a2)("typeid(", a2);
  if ( !v5 )
  {
LABEL_7:
    if ( v4 )
    {
      sub_74B930(v4, (__int64)a2);
      return ((__int64 (__fastcall *)(char *, void (__fastcall **)(const char *)))*a2)(")", a2);
    }
LABEL_9:
    sub_721090();
  }
  sub_747C50(v5, (__int64)a2);
  return ((__int64 (__fastcall *)(char *, void (__fastcall **)(const char *)))*a2)(")", a2);
}
