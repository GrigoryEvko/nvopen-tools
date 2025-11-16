// Function: sub_7503A0
// Address: 0x7503a0
//
__int64 __fastcall sub_7503A0(char *a1, __int64 a2)
{
  __int64 result; // rax
  _BYTE *v3; // rbx
  void (__fastcall *v4)(char *, __int64); // rax
  int i; // edi
  void (__fastcall *v6)(char *, __int64); // rax

  result = (unsigned int)dword_4F068C4;
  if ( dword_4F068C4 )
  {
    v3 = a1;
    if ( a1 )
    {
      (*(void (__fastcall **)(const char *))a2)(" __asm__(");
      v4 = *(void (__fastcall **)(char *, __int64))(a2 + 8);
      if ( !v4 )
        v4 = *(void (__fastcall **)(char *, __int64))a2;
      v4("\"", a2);
      for ( i = *a1; *v3; i = (char)*v3 )
      {
        ++v3;
        sub_746F50(i, a2);
      }
      v6 = *(void (__fastcall **)(char *, __int64))(a2 + 8);
      if ( !v6 )
        v6 = *(void (__fastcall **)(char *, __int64))a2;
      v6("\"", a2);
      return (*(__int64 (__fastcall **)(char *, __int64))a2)(")", a2);
    }
  }
  return result;
}
