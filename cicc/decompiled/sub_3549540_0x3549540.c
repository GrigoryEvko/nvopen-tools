// Function: sub_3549540
// Address: 0x3549540
//
void __fastcall sub_3549540(__int64 **a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-458h]
  __int64 v24; // [rsp+18h] [rbp-448h] BYREF
  __m128i v25; // [rsp+20h] [rbp-440h] BYREF
  __int64 v26[2]; // [rsp+30h] [rbp-430h] BYREF
  __int64 v27; // [rsp+40h] [rbp-420h] BYREF
  __int64 *v28; // [rsp+50h] [rbp-410h]
  __int64 v29; // [rsp+60h] [rbp-400h] BYREF
  __int64 v30[2]; // [rsp+80h] [rbp-3E0h] BYREF
  __int64 v31; // [rsp+90h] [rbp-3D0h] BYREF
  __int64 *v32; // [rsp+A0h] [rbp-3C0h]
  __int64 v33; // [rsp+B0h] [rbp-3B0h] BYREF
  void *v34; // [rsp+D0h] [rbp-390h] BYREF
  int v35; // [rsp+D8h] [rbp-388h]
  char v36; // [rsp+DCh] [rbp-384h]
  __int64 v37; // [rsp+E0h] [rbp-380h]
  __m128i v38; // [rsp+E8h] [rbp-378h]
  __int64 v39; // [rsp+F8h] [rbp-368h]
  __m128i v40; // [rsp+100h] [rbp-360h]
  __m128i v41; // [rsp+110h] [rbp-350h]
  _QWORD v42[2]; // [rsp+120h] [rbp-340h] BYREF
  _BYTE v43[324]; // [rsp+130h] [rbp-330h] BYREF
  int v44; // [rsp+274h] [rbp-1ECh]
  __int64 v45; // [rsp+278h] [rbp-1E8h]
  _QWORD v46[3]; // [rsp+280h] [rbp-1E0h] BYREF
  __m128i v47; // [rsp+298h] [rbp-1C8h]
  char *v48; // [rsp+2A8h] [rbp-1B8h]
  char *v49; // [rsp+2B0h] [rbp-1B0h]
  __int64 v50; // [rsp+2B8h] [rbp-1A8h]
  char v51; // [rsp+2C8h] [rbp-198h]
  _QWORD v52[2]; // [rsp+2D0h] [rbp-190h] BYREF
  _BYTE v53[324]; // [rsp+2E0h] [rbp-180h] BYREF
  int v54; // [rsp+424h] [rbp-3Ch]
  __int64 v55; // [rsp+428h] [rbp-38h]

  v4 = sub_B2BE50(**a1);
  if ( sub_B6EA50(v4)
    || (v21 = sub_B2BE50(**a1),
        v22 = sub_B6F970(v21),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v22 + 48LL))(v22)) )
  {
    v5 = *(_QWORD *)(a2 + 3488);
    v6 = **(_QWORD **)(v5 + 32);
    sub_2EA6600(&v24, v5);
    sub_B157E0((__int64)&v25, &v24);
    v7 = **(_QWORD **)(v6 + 32);
    v47 = _mm_loadu_si128(&v25);
    v46[2] = v7;
    v48 = "pipeliner";
    v49 = "schedule";
    v52[0] = v53;
    v46[1] = 0x200000015LL;
    v55 = v6;
    v46[0] = &unk_4A28EB8;
    v52[1] = 0x400000000LL;
    v50 = 8;
    v51 = 0;
    v53[320] = 0;
    v54 = -1;
    sub_B18290((__int64)v46, "Schedule found with Initiation Interval: ", 0x29u);
    sub_B16530(v30, "II", 2, a3[22]);
    v8 = sub_2E82FF0((__int64)v46, (__int64)v30);
    sub_B18290(v8, ", MaxStageCount: ", 0x11u);
    sub_B169E0(v26, "MaxStageCount", 13, (a3[21] - a3[20]) / a3[22]);
    v9 = sub_2E82FF0(v8, (__int64)v26);
    v13 = _mm_loadu_si128((const __m128i *)(v9 + 24));
    v14 = _mm_loadu_si128((const __m128i *)(v9 + 48));
    v15 = _mm_loadu_si128((const __m128i *)(v9 + 64));
    v35 = *(_DWORD *)(v9 + 8);
    v36 = *(_BYTE *)(v9 + 12);
    v16 = *(_QWORD *)(v9 + 16);
    v38 = v13;
    v37 = v16;
    v34 = &unk_49D9D40;
    v17 = *(_QWORD *)(v9 + 40);
    v42[1] = 0x400000000LL;
    v39 = v17;
    v42[0] = v43;
    v18 = *(unsigned int *)(v9 + 88);
    v40 = v14;
    v41 = v15;
    if ( (_DWORD)v18 )
    {
      v23 = v9;
      sub_35482E0((__int64)v42, v9 + 80, v18, v10, v11, v12);
      v9 = v23;
    }
    v43[320] = *(_BYTE *)(v9 + 416);
    v19 = *(_DWORD *)(v9 + 420);
    v20 = *(_QWORD *)(v9 + 424);
    v44 = v19;
    v45 = v20;
    v34 = &unk_4A28EB8;
    if ( v28 != &v29 )
      j_j___libc_free_0((unsigned __int64)v28);
    if ( (__int64 *)v26[0] != &v27 )
      j_j___libc_free_0(v26[0]);
    if ( v32 != &v33 )
      j_j___libc_free_0((unsigned __int64)v32);
    if ( (__int64 *)v30[0] != &v31 )
      j_j___libc_free_0(v30[0]);
    v46[0] = &unk_49D9D40;
    sub_23FD590((__int64)v52);
    if ( v24 )
      sub_B91220((__int64)&v24, v24);
    sub_2EAFC50(a1, (__int64)&v34);
    v34 = &unk_49D9D40;
    sub_23FD590((__int64)v42);
  }
}
