// Function: sub_193E1A0
// Address: 0x193e1a0
//
__int64 __fastcall sub_193E1A0(__int64 a1, _QWORD *a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v6; // r12
  __int64 v8; // rax
  int v9; // r12d
  __int64 v10; // [rsp+0h] [rbp-90h] BYREF
  _BYTE *v11; // [rsp+8h] [rbp-88h]
  _BYTE *v12; // [rsp+10h] [rbp-80h]
  __int64 v13; // [rsp+18h] [rbp-78h]
  int v14; // [rsp+20h] [rbp-70h]
  _BYTE v15[104]; // [rsp+28h] [rbp-68h] BYREF

  v6 = sub_1481F60(a2, a1, a4, a5);
  if ( sub_14562D0(v6) )
    return 0;
  if ( sub_14560B0(v6) )
    return 0;
  if ( !sub_13F9E70(a1) )
    return 0;
  v8 = sub_13F9E70(a1);
  if ( *(_BYTE *)(sub_157EBA0(v8) + 16) != 26 )
    return 0;
  v12 = v15;
  v10 = 0;
  v11 = v15;
  v13 = 8;
  v14 = 0;
  v9 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64 *, __int64))sub_3872990)(
         a3,
         v6,
         a1,
         0,
         &v10,
         1);
  if ( v12 != v11 )
    _libc_free((unsigned __int64)v12);
  return v9 ^ 1u;
}
