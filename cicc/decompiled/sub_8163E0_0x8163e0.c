// Function: sub_8163E0
// Address: 0x8163e0
//
__int64 __fastcall sub_8163E0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  _DWORD v3[5]; // [rsp+Ch] [rbp-14h] BYREF

  if ( qword_4F074B0 )
    return sub_80BC40("?", a2);
  result = sub_80C5A0(a1, 6, 0, 0, v3, a2);
  if ( !(_DWORD)result )
    return sub_80E340(a1, 0, (__int64)a2);
  return result;
}
