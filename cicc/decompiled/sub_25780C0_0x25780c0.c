// Function: sub_25780C0
// Address: 0x25780c0
//
__int64 __fastcall sub_25780C0(__m128i *a1, __int64 a2)
{
  char v2; // al
  __int64 *v3; // rsi
  __int64 v4; // r12
  unsigned __int8 *v5; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int8 *v9; // rsi
  _BYTE v10[8]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  unsigned int v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+28h] [rbp-38h]
  __int64 v15; // [rsp+30h] [rbp-30h]
  __int64 v16; // [rsp+38h] [rbp-28h]

  v2 = sub_2509800(a1);
  if ( v2 == 5 )
  {
    v4 = sub_A777F0(0xB8u, *(__int64 **)(a2 + 128));
    if ( v4 )
    {
      v7 = sub_250D070(a1);
      sub_3140AA0(v10, v7);
      v8 = sub_B491C0(v7);
      if ( v8 )
      {
        sub_3140A60(&v13, v8);
        sub_2577D20((__int64)v10, (__int64)&v13);
        v8 = 16LL * (unsigned int)v16;
        sub_C7D6A0(v14, v8, 8);
      }
      v9 = sub_250CBE0(a1->m128i_i64, v8);
      if ( v9 )
      {
        sub_3140A60(&v13, v9);
        sub_2577D20((__int64)v10, (__int64)&v13);
        sub_C7D6A0(v14, 16LL * (unsigned int)v16, 8);
      }
      v13 = 0;
      v14 = 0;
      v15 = 0;
      v16 = 0;
      sub_255D9B0((__int64)&v13, (__int64)v10);
      *(__m128i *)(v4 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v4);
      *(_BYTE *)(v4 + 96) = 0;
      *(_QWORD *)(v4 + 104) = 0;
      *(_QWORD *)v4 = &unk_4A16C58;
      *(_QWORD *)(v4 + 112) = 0;
      *(_QWORD *)(v4 + 120) = 0;
      *(_QWORD *)(v4 + 88) = &unk_4A17318;
      *(_DWORD *)(v4 + 128) = 0;
      sub_255D9B0(v4 + 104, (__int64)&v13);
      *(_BYTE *)(v4 + 136) = 1;
      *(_QWORD *)(v4 + 144) = 0;
      *(_QWORD *)(v4 + 152) = 0;
      *(_QWORD *)v4 = &unk_4A17358;
      *(_QWORD *)(v4 + 88) = &unk_4A173D8;
      *(_QWORD *)(v4 + 160) = 0;
      *(_DWORD *)(v4 + 168) = 0;
      *(_BYTE *)(v4 + 176) = 0;
      sub_C7D6A0(v14, 16LL * (unsigned int)v16, 8);
      *(_QWORD *)v4 = &unk_4A1DAD8;
      *(_QWORD *)(v4 + 88) = &unk_4A1DB60;
      sub_C7D6A0(v11, 16LL * v12, 8);
      *(_QWORD *)v4 = off_4A1DC68;
      *(_QWORD *)(v4 + 88) = &unk_4A1DCF0;
    }
    return v4;
  }
  if ( v2 > 5 )
  {
    if ( (unsigned __int8)(v2 - 6) > 1u )
      return 0;
LABEL_17:
    BUG();
  }
  if ( v2 == 4 )
  {
    v3 = *(__int64 **)(a2 + 128);
    v4 = sub_A777F0(0xB8u, v3);
    if ( v4 )
    {
      v5 = sub_250CBE0(a1->m128i_i64, (__int64)v3);
      sub_3140A60(v10, v5);
      v13 = 0;
      v14 = 0;
      v15 = 0;
      v16 = 0;
      sub_255D9B0((__int64)&v13, (__int64)v10);
      *(__m128i *)(v4 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v4);
      *(_BYTE *)(v4 + 96) = 0;
      *(_QWORD *)(v4 + 104) = 0;
      *(_QWORD *)v4 = &unk_4A16C58;
      *(_QWORD *)(v4 + 112) = 0;
      *(_QWORD *)(v4 + 120) = 0;
      *(_QWORD *)(v4 + 88) = &unk_4A17318;
      *(_DWORD *)(v4 + 128) = 0;
      sub_255D9B0(v4 + 104, (__int64)&v13);
      *(_BYTE *)(v4 + 136) = 1;
      *(_QWORD *)(v4 + 144) = 0;
      *(_QWORD *)(v4 + 152) = 0;
      *(_QWORD *)v4 = &unk_4A17358;
      *(_QWORD *)(v4 + 88) = &unk_4A173D8;
      *(_QWORD *)(v4 + 160) = 0;
      *(_DWORD *)(v4 + 168) = 0;
      *(_BYTE *)(v4 + 176) = 0;
      sub_C7D6A0(v14, 16LL * (unsigned int)v16, 8);
      *(_QWORD *)v4 = &unk_4A1DAD8;
      *(_QWORD *)(v4 + 88) = &unk_4A1DB60;
      sub_C7D6A0(v11, 16LL * v12, 8);
      *(_QWORD *)v4 = off_4A1DBA0;
      *(_QWORD *)(v4 + 88) = &unk_4A1DC28;
    }
    return v4;
  }
  if ( v2 >= 0 )
    goto LABEL_17;
  return 0;
}
