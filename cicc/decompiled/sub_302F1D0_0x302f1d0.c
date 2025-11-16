// Function: sub_302F1D0
// Address: 0x302f1d0
//
__int64 __fastcall sub_302F1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, int a7, __int64 a8)
{
  int v9; // eax
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // r14
  int v14; // r9d
  __int64 v15; // r11
  __m128i v16; // rax
  __m128i v17; // xmm0
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r14
  __m128i v23; // xmm1
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // r9d
  __int128 v28; // [rsp-20h] [rbp-D0h]
  __int128 v29; // [rsp-20h] [rbp-D0h]
  __int128 v30; // [rsp-20h] [rbp-D0h]
  __int128 v31; // [rsp-10h] [rbp-C0h]
  __int128 v32; // [rsp-10h] [rbp-C0h]
  __int64 v33; // [rsp+8h] [rbp-A8h]
  __int64 v34; // [rsp+10h] [rbp-A0h]
  int v35; // [rsp+18h] [rbp-98h]
  int v36; // [rsp+1Ch] [rbp-94h]
  __m128i v37; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v38[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v39[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v40; // [rsp+50h] [rbp-60h] BYREF
  __int64 v41; // [rsp+58h] [rbp-58h]
  __int64 v42; // [rsp+60h] [rbp-50h]
  __int64 v43; // [rsp+68h] [rbp-48h]
  __m128i v44; // [rsp+70h] [rbp-40h]

  v9 = *(_DWORD *)(a5 + 24);
  v39[0] = a1;
  v39[1] = a2;
  v38[0] = a3;
  v38[1] = a4;
  if ( v9 != 11 && v9 != 35 )
    return 0;
  v11 = *(_QWORD *)(a5 + 96);
  v12 = *(_QWORD *)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = *(_QWORD *)v12;
  *((_QWORD *)&v31 + 1) = 1;
  *(_QWORD *)&v31 = v39;
  LOWORD(v40) = 7;
  LOWORD(v42) = 7;
  v41 = 0;
  v43 = 0;
  v13 = sub_3411BE0(a8, 539, a6, (unsigned int)&v40, 2, a6, v31);
  *((_QWORD *)&v28 + 1) = 1;
  *(_QWORD *)&v28 = v38;
  LOWORD(v40) = 7;
  LOWORD(v42) = 7;
  v41 = 0;
  v43 = 0;
  v15 = sub_3411BE0(a8, 539, a6, (unsigned int)&v40, 2, v14, v28);
  if ( (a7 == 195) == (((unsigned __int8)(v12 >> 5) ^ 1) & 1) )
  {
    v34 = v13;
    v35 = 0;
    v36 = 1;
  }
  else
  {
    v34 = v15;
    v35 = 1;
    v36 = 0;
  }
  v33 = v15;
  v16.m128i_i64[0] = sub_3400BD0(a8, v12 & 0x1F, a6, 7, 0, 0, 0);
  v40 = v13;
  *((_QWORD *)&v29 + 1) = 3;
  *(_QWORD *)&v29 = &v40;
  v37 = v16;
  v17 = _mm_load_si128(&v37);
  v42 = v34;
  LODWORD(v43) = v35;
  v44 = v17;
  LODWORD(v41) = v36;
  v19 = sub_33FC220(a8, a7, a6, 7, 0, v18, v29);
  v21 = v20;
  *((_QWORD *)&v32 + 1) = 3;
  v22 = v19;
  v23 = _mm_load_si128(&v37);
  *(_QWORD *)&v32 = &v40;
  v40 = v34;
  LODWORD(v41) = v35;
  v42 = v33;
  v44 = v23;
  LODWORD(v43) = v36;
  v25 = sub_33FC220(a8, a7, a6, 7, 0, v24, v32);
  v41 = v26;
  *((_QWORD *)&v30 + 1) = 2;
  *(_QWORD *)&v30 = &v40;
  v42 = v22;
  v43 = v21;
  v40 = v25;
  return sub_33FC220(a8, 538, a6, 8, 0, v27, v30);
}
