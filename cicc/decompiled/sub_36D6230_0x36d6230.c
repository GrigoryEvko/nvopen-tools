// Function: sub_36D6230
// Address: 0x36d6230
//
void __fastcall sub_36D6230(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 **a4, __int32 a5, __int32 a6, char a7)
{
  __int64 v10; // rsi
  __int16 ***v11; // rdx
  unsigned __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // r15
  unsigned __int8 *v15; // rsi
  __int64 v16; // r15
  __int64 v17; // r9
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int32 v20; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 *v21; // [rsp+28h] [rbp-88h] BYREF
  __int64 v22[4]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v23; // [rsp+50h] [rbp-60h] BYREF
  __int64 v24; // [rsp+60h] [rbp-50h]
  __int64 v25; // [rsp+68h] [rbp-48h]
  __int64 v26; // [rsp+70h] [rbp-40h]

  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 32LL) + 56LL);
  v11 = (__int16 ***)(*(_QWORD *)(v10 + 16LL * (a5 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  v12 = *(_QWORD *)(v10 + 16LL * (a6 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v13 = *(_DWORD *)(a1 + 408) * ((__int64)(*(_QWORD *)(a1 + 368) - *(_QWORD *)(a1 + 360)) >> 3);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 392) + 16LL * (v13 + (unsigned int)*((unsigned __int16 *)*v11 + 12))) != *(_DWORD *)(*(_QWORD *)(a1 + 392) + 16LL * ((unsigned int)*(unsigned __int16 *)(*(_QWORD *)v12 + 24LL) + v13)) )
    sub_C64ED0("Copy one register into another with a different width", 1u);
  if ( v11 == &off_4A2FD40 )
  {
    v14 = -64080;
  }
  else if ( v11 == &off_4A2FCE0 )
  {
    v14 = -64000;
  }
  else if ( v11 == &off_4A2FC20 )
  {
    v14 = -64160;
    if ( (__int16 ***)v12 != v11 )
      v14 = -15120;
  }
  else if ( v11 == &off_4A2FA40 )
  {
    v14 = -64240;
    if ( (__int16 ***)v12 != v11 )
      v14 = -15200;
  }
  else if ( v11 == &off_4A2F8C0 )
  {
    v14 = -63920;
  }
  else if ( v11 == &off_4A2FB60 )
  {
    v14 = -57680;
    if ( (__int16 ***)v12 != v11 )
      v14 = -15160;
  }
  else
  {
    if ( v11 != &off_4A2F980 )
      BUG();
    v14 = -57760;
    if ( (__int16 ***)v12 != v11 )
      v14 = -15240;
  }
  v15 = *a4;
  v16 = *(_QWORD *)(a1 + 8) + v14;
  v17 = v16;
  v21 = v15;
  if ( v15 )
  {
    v20 = a5;
    sub_B96E90((__int64)&v21, (__int64)v15, 1);
    v17 = v16;
    a5 = v20;
    v22[0] = (__int64)v21;
    if ( v21 )
    {
      sub_B976B0((__int64)&v21, v21, (__int64)v22);
      a5 = v20;
      v21 = 0;
      v17 = v16;
    }
  }
  else
  {
    v22[0] = 0;
  }
  v22[1] = 0;
  v22[2] = 0;
  v18 = sub_2F26260(a2, a3, v22, v17, a5);
  v23.m128i_i64[0] = 0;
  *(__int32 *)((char *)v23.m128i_i32 + 3) = (a7 & 1) << 6;
  v24 = 0;
  *(__int32 *)((char *)v23.m128i_i32 + 2) = v23.m128i_i16[1] & 0xF00F;
  v23.m128i_i32[0] &= 0xFFF000FF;
  v23.m128i_i32[2] = a6;
  v25 = 0;
  v26 = 0;
  sub_2E8EAD0(v19, (__int64)v18, &v23);
  if ( v22[0] )
    sub_B91220((__int64)v22, v22[0]);
  if ( v21 )
    sub_B91220((__int64)&v21, (__int64)v21);
}
