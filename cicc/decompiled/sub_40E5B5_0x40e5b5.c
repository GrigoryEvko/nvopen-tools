// Function: sub_40E5B5
// Address: 0x40e5b5
//
__int64 __fastcall sub_40E5B5(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7)
{
  unsigned __int64 v10; // rdi
  int v11; // r14d
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdi
  int v16; // r9d
  unsigned __int64 v17; // rdi
  int v18; // eax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdi
  int v21; // r9d
  unsigned __int64 v22; // rdi
  int v23; // eax
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rdi
  int v26; // r9d
  unsigned __int64 v27; // rdi
  int v28; // eax
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rdi
  int v31; // r9d
  unsigned __int64 v32; // rdi
  int v33; // eax
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // rdi
  __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  __int64 result; // rax
  __int64 v43; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+28h] [rbp-98h] BYREF
  __int64 v45; // [rsp+30h] [rbp-90h] BYREF
  __int64 v46; // [rsp+38h] [rbp-88h] BYREF
  __int64 v47; // [rsp+40h] [rbp-80h] BYREF
  __int64 v48; // [rsp+48h] [rbp-78h] BYREF
  __int64 v49; // [rsp+50h] [rbp-70h] BYREF
  __int64 v50; // [rsp+58h] [rbp-68h] BYREF
  __int64 v51; // [rsp+60h] [rbp-60h] BYREF
  __int64 v52; // [rsp+68h] [rbp-58h] BYREF
  __int64 v53; // [rsp+70h] [rbp-50h] BYREF
  __int64 v54; // [rsp+78h] [rbp-48h] BYREF
  __int64 v55; // [rsp+80h] [rbp-40h] BYREF
  _QWORD v56[7]; // [rsp+88h] [rbp-38h] BYREF

  v56[0] = 7;
  v10 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v10 = sub_1313D30(v10, 0);
  if ( (unsigned int)sub_133D570(v10, a1, a2, a3, v56) )
  {
    sub_130AA40("<jemalloc>: Failure in ctl_mibnametomib()\n");
    abort();
  }
  v11 = a2 + 1;
  v12 = __readfsqword(0);
  v43 = 7;
  v44 = 8;
  *(_QWORD *)(a4 + 16) = a3;
  v13 = v12 - 2664;
  *(_DWORD *)(a5 + 8) = 5;
  if ( __readfsbyte(0xFFFFF8C8) )
    LODWORD(v13) = sub_1313D30(v13, 0);
  if ( (unsigned int)sub_133D620(
                       v13,
                       a1,
                       v11,
                       (unsigned int)"num_ops",
                       (unsigned int)&v43,
                       (int)a5 + 16,
                       (__int64)&v44,
                       0,
                       0) )
    goto LABEL_8;
  v14 = *(_QWORD *)(a5 + 16);
  *(_DWORD *)(a5 + 48) = 5;
  if ( v14 && a7 )
  {
    if ( a7 > 0x3B9AC9FF )
      v14 /= a7 / 0x3B9ACA00;
  }
  else
  {
    v14 = 0;
  }
  v15 = __readfsqword(0);
  *(_QWORD *)(a5 + 56) = v14;
  v16 = a5 + 96;
  *(_DWORD *)(a5 + 88) = 5;
  v45 = 7;
  v17 = v15 - 2664;
  v46 = 8;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v18 = sub_1313D30(v17, 0);
    v16 = a5 + 96;
    LODWORD(v17) = v18;
  }
  if ( (unsigned int)sub_133D620(v17, a1, v11, (unsigned int)"num_wait", (unsigned int)&v45, v16, (__int64)&v46, 0, 0) )
    goto LABEL_8;
  *(_DWORD *)(a5 + 128) = 5;
  v19 = *(_QWORD *)(a5 + 96);
  if ( v19 && a7 )
  {
    if ( a7 > 0x3B9AC9FF )
      v19 /= a7 / 0x3B9ACA00;
  }
  else
  {
    v19 = 0;
  }
  v20 = __readfsqword(0);
  *(_QWORD *)(a5 + 136) = v19;
  v21 = a5 + 176;
  *(_DWORD *)(a5 + 168) = 5;
  v47 = 7;
  v22 = v20 - 2664;
  v48 = 8;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v23 = sub_1313D30(v22, 0);
    v21 = a5 + 176;
    LODWORD(v22) = v23;
  }
  if ( (unsigned int)sub_133D620(
                       v22,
                       a1,
                       v11,
                       (unsigned int)"num_spin_acq",
                       (unsigned int)&v47,
                       v21,
                       (__int64)&v48,
                       0,
                       0) )
    goto LABEL_8;
  v24 = *(_QWORD *)(a5 + 176);
  *(_DWORD *)(a5 + 208) = 5;
  if ( v24 && a7 )
  {
    if ( a7 > 0x3B9AC9FF )
      v24 /= a7 / 0x3B9ACA00;
  }
  else
  {
    v24 = 0;
  }
  v25 = __readfsqword(0);
  *(_QWORD *)(a5 + 216) = v24;
  v26 = a5 + 256;
  *(_DWORD *)(a5 + 248) = 5;
  v49 = 7;
  v27 = v25 - 2664;
  v50 = 8;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v28 = sub_1313D30(v27, 0);
    v26 = a5 + 256;
    LODWORD(v27) = v28;
  }
  if ( (unsigned int)sub_133D620(
                       v27,
                       a1,
                       v11,
                       (unsigned int)"num_owner_switch",
                       (unsigned int)&v49,
                       v26,
                       (__int64)&v50,
                       0,
                       0) )
    goto LABEL_8;
  v29 = *(_QWORD *)(a5 + 256);
  *(_DWORD *)(a5 + 288) = 5;
  if ( v29 && a7 )
  {
    if ( a7 > 0x3B9AC9FF )
      v29 /= a7 / 0x3B9ACA00;
  }
  else
  {
    v29 = 0;
  }
  v30 = __readfsqword(0);
  *(_QWORD *)(a5 + 296) = v29;
  v31 = a5 + 336;
  *(_DWORD *)(a5 + 328) = 5;
  v51 = 7;
  v32 = v30 - 2664;
  v52 = 8;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v33 = sub_1313D30(v32, 0);
    v31 = a5 + 336;
    LODWORD(v32) = v33;
  }
  if ( (unsigned int)sub_133D620(
                       v32,
                       a1,
                       v11,
                       (unsigned int)"total_wait_time",
                       (unsigned int)&v51,
                       v31,
                       (__int64)&v52,
                       0,
                       0) )
    goto LABEL_8;
  v34 = *(_QWORD *)(a5 + 336);
  *(_DWORD *)(a5 + 368) = 5;
  if ( v34 && a7 )
  {
    if ( a7 > 0x3B9AC9FF )
      v34 /= a7 / 0x3B9ACA00;
  }
  else
  {
    v34 = 0;
  }
  v35 = __readfsqword(0);
  *(_QWORD *)(a5 + 376) = v34;
  v36 = a5 + 416;
  *(_DWORD *)(v36 - 8) = 5;
  v53 = 7;
  v37 = v35 - 2664;
  v54 = 8;
  if ( __readfsbyte(0xFFFFF8C8) )
    LODWORD(v37) = sub_1313D30(v37, 0);
  if ( (unsigned int)sub_133D620(
                       v37,
                       a1,
                       v11,
                       (unsigned int)"max_wait_time",
                       (unsigned int)&v53,
                       v36,
                       (__int64)&v54,
                       0,
                       0) )
    goto LABEL_8;
  v55 = 7;
  v38 = __readfsqword(0);
  v56[0] = 4;
  *(_DWORD *)(a6 + 8) = 4;
  v39 = v38 - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    LODWORD(v39) = sub_1313D30(v39, 0);
  result = sub_133D620(v39, a1, v11, (unsigned int)"max_num_thds", (unsigned int)&v55, (int)a6 + 16, (__int64)v56, 0, 0);
  if ( (_DWORD)result )
  {
LABEL_8:
    sub_130AA40("<jemalloc>: Failure in ctl_bymibname()\n");
    abort();
  }
  return result;
}
