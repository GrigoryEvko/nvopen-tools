// Function: sub_303D400
// Address: 0x303d400
//
__int64 __fastcall sub_303D400(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  int v8; // eax
  int v9; // r15d
  char v10; // r8
  int v11; // eax
  __int128 v12; // rax
  int v13; // r9d
  const __m128i *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  int v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+18h] [rbp-48h]
  __m128i v22; // [rsp+20h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v18 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v18, v6, 1);
  v7 = *(_QWORD *)(a2 + 112);
  v8 = *(_DWORD *)(a2 + 72);
  v20 = 0;
  v21 = 0;
  v22 = 0u;
  v9 = *(unsigned __int16 *)(v7 + 32);
  v19 = v8;
  v10 = sub_2EAC4F0(v7);
  v11 = 256;
  LOBYTE(v11) = v10;
  *(_QWORD *)&v12 = sub_33F1DB0(
                      a4,
                      3,
                      (unsigned int)&v18,
                      6,
                      0,
                      v11,
                      **(_QWORD **)(a2 + 40),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                      *(_OWORD *)*(_QWORD *)(a2 + 112),
                      *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
                      5,
                      0,
                      v9,
                      (__int64)&v20);
  v20 = sub_33FAF80(a4, 216, (unsigned int)&v18, 2, 0, v13, v12);
  v14 = *(const __m128i **)(a2 + 40);
  v21 = v15;
  v22 = _mm_loadu_si128(v14);
  v16 = sub_3411660(a4, &v20, 2, &v18);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v16;
}
