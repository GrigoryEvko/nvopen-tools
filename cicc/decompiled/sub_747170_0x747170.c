// Function: sub_747170
// Address: 0x747170
//
__int64 __fastcall sub_747170(char *a1, _DWORD *a2, __int64 a3)
{
  _BYTE *v4; // rbx
  void (__fastcall *v5)(char *, __int64); // rax
  int i; // edi
  void (__fastcall *v7)(char *, __int64); // rax
  __int64 result; // rax

  v4 = a1;
  if ( *a2 )
    (*(void (__fastcall **)(char *, __int64))a3)(" ", a3);
  (*(void (__fastcall **)(const char *, __int64))a3)("__attribute__((", a3);
  (*(void (__fastcall **)(const char *, __int64))a3)("__section__", a3);
  (*(void (__fastcall **)(char *, __int64))a3)("(", a3);
  v5 = *(void (__fastcall **)(char *, __int64))(a3 + 8);
  if ( !v5 )
    v5 = *(void (__fastcall **)(char *, __int64))a3;
  v5("\"", a3);
  for ( i = *a1; *v4; i = (char)*v4 )
  {
    ++v4;
    sub_746F50(i, a3);
  }
  v7 = *(void (__fastcall **)(char *, __int64))(a3 + 8);
  if ( !v7 )
    v7 = *(void (__fastcall **)(char *, __int64))a3;
  v7("\"", a3);
  result = (*(__int64 (__fastcall **)(const char *, __int64))a3)(")))", a3);
  *a2 = 1;
  return result;
}
