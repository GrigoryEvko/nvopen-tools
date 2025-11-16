// Function: sub_3038500
// Address: 0x3038500
//
__int64 __fastcall sub_3038500(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdi
  __m128i v12; // xmm0
  __int64 v13; // r14
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // r15
  __int64 v16; // rax
  int v17; // edx
  unsigned __int16 v18; // ax
  __int64 v19; // rdx
  void *v20; // r14
  int v21; // ebx
  __int64 v22; // rdx
  int v23; // r9d
  __int128 v24; // [rsp-20h] [rbp-C0h]
  __int64 v25; // [rsp+0h] [rbp-A0h] BYREF
  int v26; // [rsp+8h] [rbp-98h]
  __m128i v27; // [rsp+10h] [rbp-90h] BYREF
  void *v28; // [rsp+20h] [rbp-80h] BYREF
  __int64 v29; // [rsp+28h] [rbp-78h]
  __int64 v30; // [rsp+30h] [rbp-70h]
  __m128i v31; // [rsp+38h] [rbp-68h]
  const char *v32; // [rsp+48h] [rbp-58h]
  __int16 v33; // [rsp+68h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v25 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v25, v7, 1);
  v8 = *(__int64 **)(a4 + 40);
  v26 = *(_DWORD *)(a2 + 72);
  v9 = a1[67127];
  if ( *(_DWORD *)(v9 + 336) > 0x48u && *(_DWORD *)(v9 + 340) > 0x207u )
  {
    v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*a1 + 32LL);
    v16 = sub_2E79000(v8);
    if ( v15 == sub_2D42F30 )
    {
      v17 = sub_AE2980(v16, 5u)[1];
      v18 = 2;
      if ( v17 != 1 )
      {
        v18 = 3;
        if ( v17 != 2 )
        {
          v18 = 4;
          if ( v17 != 4 )
          {
            v18 = 5;
            if ( v17 != 8 )
            {
              v18 = 6;
              if ( v17 != 16 )
              {
                v18 = 7;
                if ( v17 != 32 )
                {
                  v18 = 8;
                  if ( v17 != 64 )
                    v18 = 9 * (v17 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v18 = v15((__int64)a1, v16, 5u);
    }
    v19 = *(_QWORD *)(a2 + 40);
    v20 = *(void **)v19;
    v21 = *(_DWORD *)(v19 + 8);
    v30 = sub_33F2D30(a4, (unsigned int)&v25, v18, 0, *(_QWORD *)(v19 + 40), *(_QWORD *)(v19 + 48), 0, 5);
    v31.m128i_i64[0] = v22;
    *((_QWORD *)&v24 + 1) = 2;
    *(_QWORD *)&v24 = &v28;
    v28 = v20;
    LODWORD(v29) = v21;
    v13 = sub_33FC220(a4, 542, (unsigned int)&v25, 1, 0, v23, v24);
  }
  else
  {
    v10 = *v8;
    sub_B157E0((__int64)&v27, &v25);
    v11 = *(_QWORD *)(a4 + 64);
    v12 = _mm_loadu_si128(&v27);
    v30 = v10;
    v29 = 24;
    v28 = &unk_49D9E88;
    v32 = "Support for stackrestore requires PTX ISA version >= 7.3 and target >= sm_52.";
    v33 = 259;
    v31 = v12;
    sub_B6EB20(v11, (__int64)&v28);
    v13 = **(_QWORD **)(a2 + 40);
  }
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v13;
}
