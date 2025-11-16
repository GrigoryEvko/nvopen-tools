// Function: sub_80F5E0
// Address: 0x80f5e0
//
__int64 __fastcall sub_80F5E0(__int64 a1, int a2, _QWORD *a3)
{
  __int64 result; // rax
  _DWORD v5[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( qword_4F074B0 )
    return sub_80BC40("?", a3);
  result = sub_80C5A0(a1, 6, 0, 0, v5, a3);
  if ( !(_DWORD)result )
    return sub_80E340(a1, a2, (__int64)a3);
  return result;
}
