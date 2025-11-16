// Function: sub_A54820
// Address: 0xa54820
//
__int64 __fastcall sub_A54820(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rax
  __int64 *v4; // rdx
  _QWORD *v5; // rdi
  __m128i *v6; // rdx
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 *v9; // rdx
  _QWORD *v10; // rdi
  _QWORD *v11; // rdi
  __int64 *v12; // rdx
  _QWORD *v13; // rdi
  __int64 *v14; // rdx
  _QWORD *v15; // rdi
  __m128i *v16; // rdx
  __m128i si128; // xmm0
  _QWORD *v18; // rdi
  __int64 v19; // rax
  _QWORD v21[3]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-48h]
  __m128i *v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+28h] [rbp-38h]
  __int64 v25; // [rsp+30h] [rbp-30h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v24 = 0x100000000LL;
  v25 = a1;
  v21[0] = &unk_49DD210;
  v21[1] = 0;
  v21[2] = 0;
  v22 = 0;
  v23 = 0;
  sub_CB5980(v21, 0, 0, 0);
  sub_904010((__int64)v21, "funcFlags: (");
  v3 = sub_904010((__int64)v21, "readNone: ");
  sub_CB59D0(v3, *a2 & 1);
  v4 = (__int64 *)v23;
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0xB )
  {
    v5 = (_QWORD *)sub_CB6200(v21, ", readOnly: ", 12);
  }
  else
  {
    v23->m128i_i32[2] = 540703084;
    v5 = v21;
    *v4 = 0x6E4F64616572202CLL;
    v23 = (__m128i *)((char *)v23 + 12);
  }
  sub_CB59D0(v5, (*a2 & 2) != 0);
  v6 = v23;
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0xC )
  {
    v7 = (_QWORD *)sub_CB6200(v21, ", noRecurse: ", 13);
  }
  else
  {
    v23->m128i_i32[2] = 979727218;
    v7 = v21;
    v6->m128i_i64[0] = 0x756365526F6E202CLL;
    v6->m128i_i8[12] = 32;
    v23 = (__m128i *)((char *)v23 + 13);
  }
  sub_CB59D0(v7, (*a2 & 4) != 0);
  v8 = sub_904010((__int64)v21, ", returnDoesNotAlias: ");
  sub_CB59D0(v8, (*a2 & 8) != 0);
  v9 = (__int64 *)v23;
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0xB )
  {
    v10 = (_QWORD *)sub_CB6200(v21, ", noInline: ", 12);
  }
  else
  {
    v23->m128i_i32[2] = 540697966;
    v10 = v21;
    *v9 = 0x696C6E496F6E202CLL;
    v23 = (__m128i *)((char *)v23 + 12);
  }
  sub_CB59D0(v10, (*a2 & 0x10) != 0);
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0xF )
  {
    v11 = (_QWORD *)sub_CB6200(v21, ", alwaysInline: ", 16);
  }
  else
  {
    v11 = v21;
    *v23++ = _mm_load_si128((const __m128i *)&xmmword_3F24AD0);
  }
  sub_CB59D0(v11, (*a2 & 0x20) != 0);
  v12 = (__int64 *)v23;
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0xB )
  {
    v13 = (_QWORD *)sub_CB6200(v21, ", noUnwind: ", 12);
  }
  else
  {
    v23->m128i_i32[2] = 540697710;
    v13 = v21;
    *v12 = 0x69776E556F6E202CLL;
    v23 = (__m128i *)((char *)v23 + 12);
  }
  sub_CB59D0(v13, (*a2 & 0x40) != 0);
  v14 = (__int64 *)v23;
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0xB )
  {
    v15 = (_QWORD *)sub_CB6200(v21, ", mayThrow: ", 12);
  }
  else
  {
    v23->m128i_i32[2] = 540702575;
    v15 = v21;
    *v14 = 0x72685479616D202CLL;
    v23 = (__m128i *)((char *)v23 + 12);
  }
  sub_CB59D0(v15, *a2 >> 7);
  v16 = v23;
  if ( (unsigned __int64)(v22 - (_QWORD)v23) <= 0x11 )
  {
    v18 = (_QWORD *)sub_CB6200(v21, ", hasUnknownCall: ", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F24AE0);
    v18 = v21;
    v23[1].m128i_i16[0] = 8250;
    *v16 = si128;
    v23 = (__m128i *)((char *)v23 + 18);
  }
  sub_CB59D0(v18, a2[1] & 1);
  v19 = sub_904010((__int64)v21, ", mustBeUnreachable: ");
  sub_CB59D0(v19, (a2[1] & 2) != 0);
  sub_904010((__int64)v21, ")");
  v21[0] = &unk_49DD210;
  sub_CB5840(v21);
  return a1;
}
