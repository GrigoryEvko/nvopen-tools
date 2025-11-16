// Function: sub_750460
// Address: 0x750460
//
__int64 __fastcall sub_750460(unsigned __int8 a1, void (__fastcall **a2)(const char *))
{
  __int64 result; // rax
  void (__fastcall *v3)(const char *); // rax
  void (__fastcall *v4)(const char *); // rax

  result = (unsigned int)dword_4F068C4;
  if ( dword_4F068C4 )
  {
    (*a2)(" __asm__(");
    v3 = a2[1];
    if ( !v3 )
      v3 = *a2;
    ((void (__fastcall *)(char *, void (__fastcall **)(const char *)))v3)("\"", a2);
    ((void (__fastcall *)(_QWORD, void (__fastcall **)(const char *)))*a2)(*(&off_4B6DCE0 + a1), a2);
    v4 = a2[1];
    if ( !v4 )
      v4 = *a2;
    ((void (__fastcall *)(char *, void (__fastcall **)(const char *)))v4)("\"", a2);
    return ((__int64 (__fastcall *)(char *, void (__fastcall **)(const char *)))*a2)(")", a2);
  }
  return result;
}
