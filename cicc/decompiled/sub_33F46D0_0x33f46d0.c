// Function: sub_33F46D0
// Address: 0x33f46d0
//
__m128i *__fastcall sub_33F46D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  int v4; // eax
  __int64 *v5; // r15
  __int64 v6; // r9
  __int64 v7; // r14
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // edx
  unsigned __int16 v13; // ax
  __m128i *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r9
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int16 *v22; // rax
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int8 v27; // al
  __m128i *v28; // r14
  unsigned __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  unsigned __int64 v32; // [rsp+10h] [rbp-90h]
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+20h] [rbp-80h] BYREF
  int v35; // [rsp+28h] [rbp-78h]
  __int128 v36; // [rsp+30h] [rbp-70h]
  __int64 v37; // [rsp+40h] [rbp-60h]
  __int64 v38; // [rsp+50h] [rbp-50h] BYREF
  __int64 v39; // [rsp+58h] [rbp-48h]
  __int64 v40; // [rsp+60h] [rbp-40h]
  __int64 v41; // [rsp+68h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v34 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v34, v3, 1);
  v4 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 **)(a2 + 40);
  BYTE4(v37) = 0;
  *((_QWORD *)&v36 + 1) = 0;
  v6 = *(_QWORD *)(a1 + 16);
  v35 = v4;
  v7 = *(_QWORD *)(v5[15] + 96);
  v8 = *(_QWORD *)(v5[20] + 96);
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  *(_QWORD *)&v36 = v8 & 0xFFFFFFFFFFFFFFFBLL;
  v9 = 0;
  if ( v8 )
  {
    v10 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
      v10 = **(_QWORD **)(v10 + 16);
    v9 = *(_DWORD *)(v10 + 8) >> 8;
  }
  LODWORD(v37) = v9;
  v31 = v6;
  v33 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v6 + 32LL);
  v11 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v33 == sub_2D42F30 )
  {
    v12 = sub_AE2980(v11, 0)[1];
    v13 = 2;
    if ( v12 != 1 )
    {
      v13 = 3;
      if ( v12 != 2 )
      {
        v13 = 4;
        if ( v12 != 4 )
        {
          v13 = 5;
          if ( v12 != 8 )
          {
            v13 = 6;
            if ( v12 != 16 )
            {
              v13 = 7;
              if ( v12 != 32 )
              {
                v13 = 8;
                if ( v12 != 64 )
                  v13 = 9 * (v12 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v13 = v33(v31, v11, 0);
  }
  v14 = sub_33F1F00((__int64 *)a1, v13, 0, (__int64)&v34, *v5, v5[1], v5[10], v5[11], v36, v37, 0, 0, (__int64)&v38, 0);
  v38 = 0;
  v16 = (__int64)v14;
  v18 = v17;
  v19 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  if ( v7 )
  {
    v20 = *(_QWORD *)(v7 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
      v20 = **(_QWORD **)(v20 + 16);
    v19 = *(_DWORD *)(v20 + 8) >> 8;
  }
  v21 = *(_QWORD *)(a2 + 40);
  LODWORD(v37) = v19;
  v22 = (unsigned __int16 *)(*(_QWORD *)(v16 + 48) + 16LL * (unsigned int)v18);
  *(_QWORD *)&v36 = v7 & 0xFFFFFFFFFFFFFFFBLL;
  v23 = *(_QWORD *)(v21 + 40);
  v24 = *(_QWORD *)(v21 + 48);
  v30 = v18;
  v25 = *((_QWORD *)v22 + 1);
  v26 = *v22;
  BYTE4(v37) = 0;
  v32 = v16;
  *((_QWORD *)&v36 + 1) = 0;
  v27 = sub_33CC4A0(a1, v26, v25, v15, v16, v18);
  v28 = sub_33F4560((_QWORD *)a1, v32, 1u, (__int64)&v34, v32, v30, v23, v24, v36, v37, v27, 0, (__int64)&v38);
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
  return v28;
}
