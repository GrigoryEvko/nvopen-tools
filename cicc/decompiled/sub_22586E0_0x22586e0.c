// Function: sub_22586E0
// Address: 0x22586e0
//
__int64 __fastcall sub_22586E0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  void *v4; // rdx
  _QWORD *v5; // rdi
  __int64 v6; // rdi
  _BYTE *v7; // rax
  __int64 v8; // rax
  __m128i *v9; // rdx
  __int64 v10; // rdi
  __m128i si128; // xmm0
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // rax
  _QWORD v17[3]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  _BYTE *v19; // [rsp+20h] [rbp-40h]
  __int64 v20; // [rsp+28h] [rbp-38h]
  __int64 v21; // [rsp+30h] [rbp-30h]

  v20 = 0x100000000LL;
  v21 = a1 + 80;
  v17[1] = 0;
  v17[0] = &unk_49DD210;
  v17[2] = 0;
  v18 = 0;
  v19 = 0;
  sub_CB5980((__int64)v17, 0, 0, 0);
  v4 = v19;
  if ( (unsigned __int64)(v18 - (_QWORD)v19) <= 0xA )
  {
    v5 = (_QWORD *)sub_CB6200((__int64)v17, "IR version ", 0xBu);
  }
  else
  {
    v19[10] = 32;
    v5 = v17;
    qmemcpy(v4, "IR version", 10);
    v19 += 11;
  }
  v6 = sub_CB59D0((__int64)v5, a2);
  v7 = *(_BYTE **)(v6 + 32);
  if ( *(_BYTE **)(v6 + 24) == v7 )
  {
    v6 = sub_CB6200(v6, (unsigned __int8 *)".", 1u);
  }
  else
  {
    *v7 = 46;
    ++*(_QWORD *)(v6 + 32);
  }
  v8 = sub_CB59D0(v6, a3);
  v9 = *(__m128i **)(v8 + 32);
  v10 = v8;
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 0x22u )
  {
    v10 = sub_CB6200(v8, " incompatible with current version ", 0x23u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4281860);
    v9[2].m128i_i8[2] = 32;
    v9[2].m128i_i16[0] = 28271;
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_4281870);
    *(_QWORD *)(v8 + 32) += 35LL;
  }
  v12 = sub_CB59F0(v10, 2);
  v13 = *(_BYTE **)(v12 + 32);
  if ( *(_BYTE **)(v12 + 24) == v13 )
  {
    v12 = sub_CB6200(v12, (unsigned __int8 *)".", 1u);
  }
  else
  {
    *v13 = 46;
    ++*(_QWORD *)(v12 + 32);
  }
  v14 = sub_CB59F0(v12, 0);
  v15 = *(_BYTE **)(v14 + 32);
  if ( *(_BYTE **)(v14 + 24) == v15 )
  {
    sub_CB6200(v14, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v15 = 10;
    ++*(_QWORD *)(v14 + 32);
  }
  v17[0] = &unk_49DD210;
  sub_CB5840((__int64)v17);
  return 0;
}
