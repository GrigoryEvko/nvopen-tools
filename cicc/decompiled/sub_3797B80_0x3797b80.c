// Function: sub_3797B80
// Address: 0x3797b80
//
unsigned __int8 *__fastcall sub_3797B80(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r14d
  const __m128i *v5; // rax
  __int64 v6; // rsi
  __m128i v7; // xmm0
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // rax
  int v16; // edx
  unsigned __int16 v17; // ax
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r11
  unsigned __int8 *v23; // r10
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // r15
  unsigned __int16 *v33; // rdx
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // r8
  unsigned __int8 *v37; // r14
  unsigned int v39; // edx
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int128 v42; // [rsp-10h] [rbp-C0h]
  __int64 v43; // [rsp-8h] [rbp-B8h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v45)(__int64, __int64, __int64, __int64); // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v47; // [rsp+10h] [rbp-A0h]
  __int64 v48; // [rsp+18h] [rbp-98h]
  __int64 v49; // [rsp+20h] [rbp-90h]
  __int128 v50; // [rsp+30h] [rbp-80h]
  unsigned __int16 v51; // [rsp+40h] [rbp-70h] BYREF
  __int64 v52; // [rsp+48h] [rbp-68h]
  __int64 v53; // [rsp+50h] [rbp-60h] BYREF
  int v54; // [rsp+58h] [rbp-58h]
  __int16 v55; // [rsp+60h] [rbp-50h] BYREF
  __int64 v56; // [rsp+68h] [rbp-48h]

  v5 = *(const __m128i **)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = _mm_loadu_si128(v5);
  v8 = *(_QWORD *)(v5->m128i_i64[0] + 48) + 16LL * v5->m128i_u32[2];
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v53 = v6;
  v51 = v9;
  v52 = v10;
  if ( v6 )
  {
    sub_B96E90((__int64)&v53, v6, 1);
    v10 = v52;
    v9 = v51;
  }
  v11 = *a1;
  v54 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)&v55, v11, *(_QWORD *)(a1[1] + 64), v9, v10);
  if ( (_BYTE)v55 == 5 )
  {
    v27 = v7.m128i_i64[0];
    v28 = sub_37946F0((__int64)a1, v7.m128i_u64[0], v7.m128i_i64[1]);
    v32 = v39;
  }
  else
  {
    v12 = a1[1];
    v49 = *a1;
    v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 72LL);
    v13 = sub_2E79000(*(__int64 **)(v12 + 40));
    v14 = v13;
    if ( v45 == sub_2FE4D20 )
    {
      v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v49 + 32LL);
      if ( v15 == sub_2D42F30 )
      {
        v16 = sub_AE2980(v14, 0)[1];
        v17 = 2;
        if ( v16 != 1 )
        {
          v17 = 3;
          if ( v16 != 2 )
          {
            v17 = 4;
            if ( v16 != 4 )
            {
              v17 = 5;
              if ( v16 != 8 )
              {
                v17 = 6;
                if ( v16 != 16 )
                {
                  v17 = 7;
                  if ( v16 != 32 )
                  {
                    v17 = 8;
                    if ( v16 != 64 )
                      v17 = 9 * (v16 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v17 = v15(v49, v14, 0);
      }
    }
    else
    {
      v17 = ((__int64 (__fastcall *)(__int64, __int64))v45)(v49, v13);
    }
    v18 = sub_3400BD0(v12, 0, (__int64)&v53, v17, 0, 0, v7, 0);
    v22 = v19;
    v23 = v18;
    if ( v51 )
    {
      v24 = 0;
      LOWORD(v25) = word_4456580[v51 - 1];
    }
    else
    {
      v47 = v18;
      v48 = v19;
      v25 = sub_3009970((__int64)&v51, 0, 0, v43, v20);
      v23 = v47;
      v22 = v48;
      v44 = v25;
      v24 = v41;
    }
    *((_QWORD *)&v42 + 1) = v22;
    v26 = v44;
    *(_QWORD *)&v42 = v23;
    v27 = 158;
    LOWORD(v26) = v25;
    v28 = (__int64)sub_3406EB0((_QWORD *)v12, 0x9Eu, (__int64)&v53, v26, v24, v21, *(_OWORD *)&v7, v42);
    v32 = v31;
  }
  v33 = *(unsigned __int16 **)(a2 + 48);
  v34 = *v33;
  v35 = *((_QWORD *)v33 + 1);
  v55 = v34;
  v56 = v35;
  if ( (_WORD)v34 )
  {
    v36 = 0;
    LOWORD(v34) = word_4456580[v34 - 1];
  }
  else
  {
    v46 = v28;
    v34 = sub_3009970((__int64)&v55, v27, v35, v28, v29);
    v28 = v46;
    HIWORD(v2) = HIWORD(v34);
    v36 = v40;
  }
  LOWORD(v2) = v34;
  *(_QWORD *)&v50 = v28;
  *((_QWORD *)&v50 + 1) = v32 | v7.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v37 = sub_3406EB0(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v53,
          v2,
          v36,
          v30,
          v50,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
  return v37;
}
