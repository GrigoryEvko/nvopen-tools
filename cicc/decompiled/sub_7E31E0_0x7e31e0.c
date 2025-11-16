// Function: sub_7E31E0
// Address: 0x7e31e0
//
_DWORD *__fastcall sub_7E31E0(__int64 a1)
{
  _DWORD *result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  _QWORD v5[29]; // [rsp-E8h] [rbp-E8h] BYREF

  result = &dword_4F077C4;
  if ( dword_4F077C4 == 2 && (*(_BYTE *)(a1 + 171) & 8) == 0 )
  {
    sub_76C7C0((__int64)v5);
    v5[2] = sub_7E4690;
    return (_DWORD *)sub_76D560(a1, (__int64)v5, v2, v3, v4);
  }
  return result;
}
