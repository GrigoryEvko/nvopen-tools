// Function: sub_D90370
// Address: 0xd90370
//
__int64 __fastcall sub_D90370(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  _QWORD v6[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 (__fastcall *v7)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-70h]
  __int64 (__fastcall *v8)(__int64 *, unsigned __int64); // [rsp+18h] [rbp-68h]
  __m128i v9[6]; // [rsp+20h] [rbp-60h] BYREF

  v2 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8780C);
  if ( v2 && (v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F8780C)) != 0 )
    v4 = *(_QWORD *)(v3 + 176);
  else
    v4 = 0;
  v6[0] = a1;
  v8 = sub_D856A0;
  v7 = sub_D85850;
  sub_D90260(v9[0].m128i_i64, a2, (__int64)v6, v4);
  sub_D89CB0(a1 + 176, v9);
  sub_D89DE0((__int64)v9, (__int64)v9);
  if ( v7 )
    v7(v6, v6, 3);
  return 0;
}
