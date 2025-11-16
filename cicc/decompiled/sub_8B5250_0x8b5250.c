// Function: sub_8B5250
// Address: 0x8b5250
//
_BOOL8 __fastcall sub_8B5250(__int64 *a1, __m128i *a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  _QWORD *v5; // r13
  __m128i *v6; // rax
  __int64 v7; // rsi
  _BOOL4 v8; // r13d
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r8
  __m128i *v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *a1;
  v4 = *(_QWORD *)(*a1 + 88);
  v5 = **(_QWORD ***)(v4 + 32);
  v13[0] = 0;
  v6 = sub_8A3C00((__int64)v5, 0, 0, 0);
  v7 = a1[24];
  v13[0] = v6;
  if ( (unsigned int)sub_8B3500(a2, v7, (__int64 *)v13, (__int64)v5, 0x400u)
    && (unsigned int)sub_8AF210(v13[0], v5, 0, v3, v4, 0) )
  {
    if ( !HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4 || (v8 = 1, !qword_4F077A8) )
    {
      v8 = 1;
      v11 = sub_8A0370(v3, v13, 0, 0, 0, 0, 0)[11];
      if ( a2 != (__m128i *)v11 )
        v8 = sub_8D97D0(a2, v11, 0, v10, v12) != 0;
    }
  }
  else
  {
    v8 = 0;
  }
  sub_725130(v13[0]->m128i_i64);
  return v8;
}
