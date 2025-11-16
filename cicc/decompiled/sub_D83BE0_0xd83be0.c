// Function: sub_D83BE0
// Address: 0xd83be0
//
__int64 __fastcall sub_D83BE0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  _QWORD v9[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 (__fastcall *v10)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-60h]
  __int64 (__fastcall *v11)(__int64 *, __int64); // [rsp+18h] [rbp-58h]
  __int64 v12; // [rsp+20h] [rbp-50h] BYREF
  char v13; // [rsp+28h] [rbp-48h]
  __int64 (__fastcall *v14)(const __m128i **, const __m128i *, int); // [rsp+30h] [rbp-40h]
  __int64 (__fastcall *v15)(__int64, __int64); // [rsp+38h] [rbp-38h]

  v6 = sub_BC0510(a4, &unk_4F87C68, (__int64)a3);
  v7 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  v13 = sub_D89FA0(a3);
  v15 = sub_D76BB0;
  v12 = v7;
  v14 = sub_D76060;
  v9[0] = v7;
  v11 = sub_D75D20;
  v10 = sub_D76090;
  sub_D81040(a1, a3, (__int64)v9, v6 + 8, (__int64)&v12);
  if ( v10 )
    v10(v9, v9, 3);
  if ( v14 )
    v14((const __m128i **)&v12, (const __m128i *)&v12, 3);
  return a1;
}
