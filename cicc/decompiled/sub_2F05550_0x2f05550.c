// Function: sub_2F05550
// Address: 0x2f05550
//
__int64 __fastcall sub_2F05550(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  void *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v13; // [rsp+8h] [rbp-358h]
  _QWORD v14[10]; // [rsp+10h] [rbp-350h] BYREF
  __int16 v15; // [rsp+60h] [rbp-300h]
  char v16; // [rsp+62h] [rbp-2FEh]
  __int64 v17; // [rsp+68h] [rbp-2F8h]
  __int64 v18; // [rsp+70h] [rbp-2F0h]
  __int64 v19; // [rsp+78h] [rbp-2E8h]
  char *v20; // [rsp+80h] [rbp-2E0h]
  __int64 v21; // [rsp+88h] [rbp-2D8h]
  int v22; // [rsp+90h] [rbp-2D0h]
  char v23; // [rsp+94h] [rbp-2CCh]
  char v24; // [rsp+98h] [rbp-2C8h] BYREF
  char *v25; // [rsp+D8h] [rbp-288h]
  __int64 v26; // [rsp+E0h] [rbp-280h]
  char v27; // [rsp+E8h] [rbp-278h] BYREF
  int v28; // [rsp+118h] [rbp-248h]
  __int64 v29; // [rsp+120h] [rbp-240h]
  __int64 v30; // [rsp+128h] [rbp-238h]
  __int64 v31; // [rsp+130h] [rbp-230h]
  __int64 v32; // [rsp+138h] [rbp-228h]
  char *v33; // [rsp+140h] [rbp-220h]
  __int64 v34; // [rsp+148h] [rbp-218h]
  char v35; // [rsp+150h] [rbp-210h] BYREF
  char *v36; // [rsp+190h] [rbp-1D0h]
  __int64 v37; // [rsp+198h] [rbp-1C8h]
  char v38; // [rsp+1A0h] [rbp-1C0h] BYREF
  char *v39; // [rsp+1E0h] [rbp-180h]
  __int64 v40; // [rsp+1E8h] [rbp-178h]
  char v41; // [rsp+1F0h] [rbp-170h] BYREF
  char *v42; // [rsp+230h] [rbp-130h]
  __int64 v43; // [rsp+238h] [rbp-128h]
  char v44; // [rsp+240h] [rbp-120h] BYREF
  __int64 v45; // [rsp+260h] [rbp-100h]
  __int64 v46; // [rsp+268h] [rbp-F8h]
  __int64 v47; // [rsp+270h] [rbp-F0h]
  __int64 v48; // [rsp+278h] [rbp-E8h]
  int v49; // [rsp+280h] [rbp-E0h]
  __int64 v50; // [rsp+288h] [rbp-D8h]
  __int64 v51; // [rsp+290h] [rbp-D0h]
  __int64 v52; // [rsp+298h] [rbp-C8h]
  __int64 v53; // [rsp+2A0h] [rbp-C0h]
  int v54; // [rsp+2A8h] [rbp-B8h]
  char v55; // [rsp+2ACh] [rbp-B4h]
  char *v56; // [rsp+2B0h] [rbp-B0h]
  __int64 v57; // [rsp+2B8h] [rbp-A8h]
  char v58; // [rsp+2C0h] [rbp-A0h] BYREF
  char *v59; // [rsp+2C8h] [rbp-98h]
  __int64 v60; // [rsp+2D0h] [rbp-90h]
  char v61; // [rsp+2D8h] [rbp-88h] BYREF
  __int64 v62; // [rsp+310h] [rbp-50h]
  __int64 v63; // [rsp+318h] [rbp-48h]
  char v64; // [rsp+320h] [rbp-40h]
  __int64 v65; // [rsp+324h] [rbp-3Ch]

  if ( (*(_BYTE *)(a3 + 345) & 2) == 0 )
  {
    v6 = sub_CB72A0();
    v14[0] = a4;
    v14[1] = 0;
    v7 = *a2;
    if ( !v6 )
    {
      v13 = *a2;
      v6 = sub_CB7330();
      v7 = v13;
    }
    v14[2] = v6;
    v15 = 0;
    v20 = &v24;
    v25 = &v27;
    v33 = &v35;
    v34 = 0x1000000000LL;
    v37 = 0x1000000000LL;
    v40 = 0x1000000000LL;
    v42 = &v44;
    v14[3] = v7;
    v36 = &v38;
    v43 = 0x400000000LL;
    v26 = 0x600000000LL;
    v39 = &v41;
    memset(&v14[4], 0, 48);
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v21 = 8;
    v22 = 0;
    v23 = 1;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v45 = 0;
    v56 = &v58;
    v57 = 0x100000000LL;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 1;
    v59 = &v61;
    v60 = 0x600000000LL;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    sub_2F02A10((__int64)v14, (const char *)a3);
    sub_2EF2DE0((__int64)v14, a3, v8, v9, v10, v11);
  }
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
