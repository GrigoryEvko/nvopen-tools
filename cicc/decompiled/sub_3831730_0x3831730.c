// Function: sub_3831730
// Address: 0x3831730
//
unsigned __int8 *__fastcall sub_3831730(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // rax
  _DWORD *v12; // rax
  __int64 v13; // r10
  unsigned __int16 v14; // r8
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int8 *v18; // rax
  _QWORD *v19; // r9
  unsigned __int8 *v20; // r10
  unsigned __int16 *v21; // rax
  __int64 v22; // r11
  __int64 v23; // r8
  __int64 v24; // rdx
  unsigned __int8 *v25; // rax
  unsigned int v26; // edx
  unsigned __int8 *v27; // r12
  bool v29; // al
  __int64 v30; // rcx
  __int16 v31; // ax
  unsigned __int16 v32; // ax
  unsigned __int16 v33; // ax
  __int128 v34; // [rsp-20h] [rbp-B0h]
  __int128 v35; // [rsp-10h] [rbp-A0h]
  unsigned __int8 *v36; // [rsp+0h] [rbp-90h]
  __int64 v37; // [rsp+8h] [rbp-88h]
  unsigned int v38; // [rsp+1Ch] [rbp-74h]
  __int64 (__fastcall *v39)(__int64, __int64, __int64, __int64); // [rsp+20h] [rbp-70h]
  _QWORD *v40; // [rsp+20h] [rbp-70h]
  __int64 v41; // [rsp+30h] [rbp-60h]
  __int64 v42; // [rsp+30h] [rbp-60h]
  __int64 v43; // [rsp+38h] [rbp-58h]
  __int64 v44; // [rsp+40h] [rbp-50h] BYREF
  int v45; // [rsp+48h] [rbp-48h]
  __int16 v46; // [rsp+50h] [rbp-40h] BYREF
  __int64 v47; // [rsp+58h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v44 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v44, v5, 1);
  v45 = *(_DWORD *)(a2 + 72);
  v6 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v8 = v7;
  v41 = *a1;
  v43 = a1[1];
  v39 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 72LL);
  v9 = sub_2E79000(*(__int64 **)(v43 + 40));
  v10 = v9;
  if ( v39 == sub_2FE4D20 )
  {
    v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v41 + 32LL);
    if ( v11 == sub_2D42F30 )
    {
      v12 = sub_AE2980(v10, 0);
      v13 = v43;
      v14 = 2;
      v15 = v12[1];
      if ( v15 != 1 )
      {
        v14 = 3;
        if ( v15 != 2 )
        {
          v14 = 4;
          if ( v15 != 4 )
          {
            v14 = 5;
            if ( v15 != 8 )
            {
              v14 = 6;
              if ( v15 != 16 )
              {
                v14 = 7;
                if ( v15 != 32 )
                {
                  v14 = 8;
                  if ( v15 != 64 )
                    v14 = 9 * (v15 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v33 = v11(v41, v10, 0);
      v13 = v43;
      v14 = v33;
    }
  }
  else
  {
    v32 = ((__int64 (__fastcall *)(__int64, __int64))v39)(v41, v9);
    v13 = v43;
    v14 = v32;
  }
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_QWORD *)(v16 + 40);
  v18 = sub_33FB310(v13, v17, *(_QWORD *)(v16 + 48), (__int64)&v44, v14, 0, a3);
  v19 = (_QWORD *)a1[1];
  v20 = v18;
  v21 = *(unsigned __int16 **)(v6 + 48);
  v22 = v24;
  v23 = *((_QWORD *)v21 + 1);
  LODWORD(v24) = *v21;
  v47 = v23;
  v46 = v24;
  if ( (_WORD)v24 )
  {
    if ( (unsigned __int16)(v24 - 17) <= 0xD3u )
    {
      v23 = 0;
      LOWORD(v24) = word_4456580[(int)v24 - 1];
    }
  }
  else
  {
    v36 = v20;
    v37 = v22;
    v38 = v24;
    v42 = v23;
    v40 = v19;
    v29 = sub_30070B0((__int64)&v46);
    v19 = v40;
    v23 = v42;
    LOWORD(v24) = v38;
    v20 = v36;
    v22 = v37;
    if ( v29 )
    {
      v31 = sub_3009970((__int64)&v46, v17, v38, v30, v42);
      v20 = v36;
      v22 = v37;
      v23 = v24;
      v19 = v40;
      LOWORD(v24) = v31;
    }
  }
  *((_QWORD *)&v35 + 1) = v22;
  *(_QWORD *)&v35 = v20;
  *((_QWORD *)&v34 + 1) = v8;
  *(_QWORD *)&v34 = v6;
  v25 = sub_3406EB0(v19, 0x9Eu, (__int64)&v44, (unsigned __int16)v24, v23, (__int64)v19, v34, v35);
  v27 = sub_33FAFB0(
          a1[1],
          (__int64)v25,
          v26,
          (__int64)&v44,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          a3);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v27;
}
