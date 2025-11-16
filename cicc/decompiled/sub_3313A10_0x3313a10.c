// Function: sub_3313A10
// Address: 0x3313a10
//
__int64 __fastcall sub_3313A10(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  __int32 v8; // edx
  __int32 v9; // esi
  __int64 v10; // rax
  char v12; // al
  __int64 v13; // r14
  __int64 v14; // rsi
  __int64 v15; // r11
  _QWORD *v16; // rcx
  __m128i v17; // xmm0
  __int64 v18; // r15
  int v19; // r9d
  __int16 v20; // bx
  unsigned __int16 *v21; // rsi
  __int64 v22; // r14
  __int32 v23; // edx
  __int32 v24; // ebx
  _QWORD *v25; // [rsp+0h] [rbp-90h]
  int v26; // [rsp+Ch] [rbp-84h]
  int v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  int v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  int v31; // [rsp+38h] [rbp-58h]
  __m128i v32; // [rsp+40h] [rbp-50h] BYREF
  __m128i v33; // [rsp+50h] [rbp-40h]

  v2 = a2;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(v3 + 120);
  v28 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v28, v4, 1);
  v29 = *(_DWORD *)(v2 + 72);
  if ( (unsigned __int8)sub_33D1AE0(v5, 0) )
  {
    v6 = *(_QWORD *)(v2 + 40);
    v7 = *(_QWORD *)v6;
    v8 = *(_DWORD *)(v6 + 8);
    v9 = *(_DWORD *)(v6 + 168);
    v10 = *(_QWORD *)(v6 + 160);
    v33.m128i_i64[0] = v7;
    v32.m128i_i64[0] = v10;
    v32.m128i_i32[2] = v9;
    v33.m128i_i32[2] = v8;
LABEL_5:
    v2 = sub_32EB790(a1, v2, v32.m128i_i64, 2, 1);
    goto LABEL_6;
  }
  if ( (unsigned __int8)sub_33D1720(v5, 0) )
  {
    if ( (*(_WORD *)(v2 + 32) & 0x380) == 0 )
    {
      v12 = *(_BYTE *)(v2 + 33);
      if ( (v12 & 0x10) == 0 && (v12 & 0xC) == 0 )
      {
        v13 = *(_QWORD *)(v2 + 112);
        v14 = *(_QWORD *)(v2 + 80);
        v15 = *(_QWORD *)a1;
        v16 = *(_QWORD **)(v2 + 40);
        v17 = _mm_loadu_si128((const __m128i *)(v13 + 40));
        v18 = *(_QWORD *)(v13 + 72);
        v30 = v14;
        v32 = v17;
        v33 = _mm_loadu_si128((const __m128i *)(v13 + 56));
        LOBYTE(v20) = *(_BYTE *)(v13 + 34);
        v19 = *(unsigned __int16 *)(v13 + 32);
        HIBYTE(v20) = 1;
        if ( v14 )
        {
          v27 = v15;
          v25 = v16;
          v26 = *(unsigned __int16 *)(v13 + 32);
          sub_B96E90((__int64)&v30, v14, 1);
          v16 = v25;
          v19 = v26;
          LODWORD(v15) = v27;
        }
        v21 = *(unsigned __int16 **)(v2 + 48);
        v31 = *(_DWORD *)(v2 + 72);
        v22 = sub_33F1F00(
                v15,
                *v21,
                *((_QWORD *)v21 + 1),
                (unsigned int)&v30,
                *v16,
                v16[1],
                v16[5],
                v16[6],
                *(_OWORD *)v13,
                *(_QWORD *)(v13 + 16),
                v20,
                v19,
                (__int64)&v32,
                v18);
        v24 = v23;
        if ( v30 )
          sub_B91220((__int64)&v30, v30);
        v32.m128i_i64[0] = v22;
        v32.m128i_i32[2] = v24;
        v33.m128i_i64[0] = v22;
        v33.m128i_i32[2] = 1;
        goto LABEL_5;
      }
    }
  }
  if ( *(int *)(a1 + 24) <= 2
    || !(unsigned __int8)sub_3312A90((__int64 *)a1, v2)
    && (*(int *)(a1 + 24) <= 2 || !(unsigned __int8)sub_3312210((__int64 *)a1, v2)) )
  {
    v2 = 0;
  }
LABEL_6:
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v2;
}
