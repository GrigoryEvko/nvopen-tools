// Function: sub_3038700
// Address: 0x3038700
//
__int64 __fastcall sub_3038700(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __m128i v12; // xmm0
  __m128i v13; // rax
  __int64 v14; // r14
  __int64 v16; // rax
  int v17; // edx
  int v18; // r9d
  __int16 v19; // ax
  __m128i v20; // xmm2
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int128 v24; // [rsp-10h] [rbp-D0h]
  __int64 v25; // [rsp+8h] [rbp-B8h]
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-B8h]
  __int64 v27; // [rsp+8h] [rbp-B8h]
  __int64 v28; // [rsp+10h] [rbp-B0h] BYREF
  int v29; // [rsp+18h] [rbp-A8h]
  __m128i v30[2]; // [rsp+20h] [rbp-A0h] BYREF
  void *v31; // [rsp+40h] [rbp-80h] BYREF
  __int64 v32; // [rsp+48h] [rbp-78h]
  __int64 v33; // [rsp+50h] [rbp-70h]
  __m128i v34; // [rsp+58h] [rbp-68h]
  const char *v35; // [rsp+68h] [rbp-58h]
  __int16 v36; // [rsp+88h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v28 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v28, v8, 1);
  v9 = *(__int64 **)(a4 + 40);
  v29 = *(_DWORD *)(a2 + 72);
  v10 = a1[67127];
  if ( *(_DWORD *)(v10 + 336) > 0x48u && *(_DWORD *)(v10 + 340) > 0x207u )
  {
    v26 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*a1 + 32LL);
    v16 = sub_2E79000(v9);
    if ( v26 == sub_2D42F30 )
    {
      v17 = sub_AE2980(v16, 5u)[1];
      v19 = 2;
      if ( v17 != 1 )
      {
        v19 = 3;
        if ( v17 != 2 )
        {
          v19 = 4;
          if ( v17 != 4 )
          {
            v19 = 5;
            if ( v17 != 8 )
            {
              v19 = 6;
              if ( v17 != 16 )
              {
                v19 = 7;
                if ( v17 != 32 )
                {
                  v19 = 8;
                  if ( v17 != 64 )
                    v19 = 9 * (v17 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v19 = v26((__int64)a1, v16, 5u);
    }
    v20 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    *((_QWORD *)&v24 + 1) = 1;
    *(_QWORD *)&v24 = v30;
    LOWORD(v31) = v19;
    v30[0] = v20;
    v32 = 0;
    LOWORD(v33) = 1;
    v34.m128i_i64[0] = 0;
    v27 = sub_3411BE0(a4, 543, (unsigned int)&v28, (unsigned int)&v31, 2, v18, v24);
    v22 = sub_33F2D30(
            a4,
            (unsigned int)&v28,
            *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3),
            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3 + 8),
            v27,
            v21,
            5,
            0);
    v32 = v23;
    v31 = (void *)v22;
    v33 = v27;
    v34.m128i_i32[0] = 1;
    v14 = sub_3411660(a4, &v31, 2, &v28);
  }
  else
  {
    v25 = *v9;
    sub_B157E0((__int64)v30, &v28);
    v11 = *(_QWORD *)(a4 + 64);
    v12 = _mm_loadu_si128(v30);
    v36 = 259;
    v33 = v25;
    v34 = v12;
    v32 = 24;
    v31 = &unk_49D9E88;
    v35 = "Support for stacksave requires PTX ISA version >= 7.3 and target >= sm_52.";
    sub_B6EB20(v11, (__int64)&v31);
    v13.m128i_i64[0] = sub_3400BD0(
                         a4,
                         0,
                         (unsigned int)&v28,
                         *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3),
                         *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3 + 8),
                         0,
                         0);
    v30[0] = v13;
    v30[1] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v14 = sub_3411660(a4, v30, 2, &v28);
  }
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v14;
}
