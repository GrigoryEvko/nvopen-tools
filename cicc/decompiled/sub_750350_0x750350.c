// Function: sub_750350
// Address: 0x750350
//
__int64 __fastcall sub_750350(__int64 a1, unsigned int a2, __int64 a3)
{
  bool v3; // zf
  unsigned int v5; // [rsp+Ch] [rbp-4h] BYREF

  v3 = *(_BYTE *)(a3 + 136) == 0;
  v5 = a2;
  if ( !v3 && !dword_4F068C4 || (*(_BYTE *)(a1 + 91) & 4) == 0 )
    return v5;
  sub_7450F0("__unused__", &v5, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  return v5;
}
