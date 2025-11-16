// Function: sub_5FB400
// Address: 0x5fb400
//
__int64 __fastcall sub_5FB400(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, const __m128i *a5)
{
  __int64 v7; // r15
  char v8; // cl
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v13; // r15
  __int64 v14; // rdx
  char v15; // al
  __int16 v18; // [rsp+1Eh] [rbp-232h] BYREF
  _BYTE v19[84]; // [rsp+20h] [rbp-230h] BYREF
  _BOOL4 v20; // [rsp+74h] [rbp-1DCh]
  _BOOL4 v21; // [rsp+78h] [rbp-1D8h]
  __int16 *v22; // [rsp+D8h] [rbp-178h]
  _QWORD *v23; // [rsp+E0h] [rbp-170h]
  __int64 v24; // [rsp+F8h] [rbp-158h]
  __int64 v25; // [rsp+170h] [rbp-E0h]

  v18 = 75;
  v7 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 16) + 248LL) + 88LL);
  sub_89EF00(v19, a3);
  v22 = &v18;
  v8 = *(_BYTE *)(v7 + 160);
  v21 = (v8 & 0x10) != 0;
  v20 = (v8 & 8) != 0;
  sub_860400(v23, 0);
  v9 = (_QWORD *)sub_89EE30(**(_QWORD **)(v7 + 328));
  *v23 = v9;
  v23[9] = *(_QWORD *)(*(_QWORD *)(v7 + 328) + 72LL);
  ++v24;
  v10 = v25;
  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 616) = v19;
  sub_5FA450(a4, v9, v10, a5, a2, (__int64)a3);
  v11 = *a3;
  if ( *(_BYTE *)(v11 + 80) == 20 )
  {
    v13 = *(_QWORD *)(v11 + 88);
    sub_897580(v19, v11, v13);
    v14 = *(_QWORD *)(v13 + 176);
    *(_BYTE *)(v14 + 193) |= 0x10u;
    v15 = (2 * *(_BYTE *)(a1 + 25)) & 8 | *(_BYTE *)(v14 + 198) & 0xF7;
    *(_BYTE *)(v14 + 198) = v15;
    *(_BYTE *)(v14 + 198) = (8 * *(_BYTE *)(a1 + 25)) & 0x10 | v15 & 0xEF;
    *(_QWORD *)(v13 + 328) = v23;
    sub_89F220(v19, a5, v11);
  }
  return sub_863FC0();
}
