// Function: sub_1436210
// Address: 0x1436210
//
__int64 __fastcall sub_1436210(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // ecx
  __int64 v18; // rdx
  __int64 v19; // rax
  int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // ecx
  __int64 v24; // rax
  unsigned int v25; // r13d
  void *v26; // rax
  unsigned int v27; // r13d
  int v29; // edx
  void *v30; // rsi
  int v31; // eax
  size_t v32; // rdx
  _QWORD v33[2]; // [rsp+0h] [rbp-1D0h] BYREF
  __int64 (__fastcall *v34)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-1C0h]
  __int64 (__fastcall *v35)(_QWORD *, __int64); // [rsp+18h] [rbp-1B8h]
  _BYTE v36[8]; // [rsp+20h] [rbp-1B0h] BYREF
  int v37; // [rsp+28h] [rbp-1A8h] BYREF
  __int64 v38; // [rsp+30h] [rbp-1A0h]
  int *v39; // [rsp+38h] [rbp-198h]
  int *v40; // [rsp+40h] [rbp-190h]
  __int64 v41; // [rsp+48h] [rbp-188h]
  __int64 v42; // [rsp+50h] [rbp-180h]
  __int64 v43; // [rsp+58h] [rbp-178h]
  __int64 v44; // [rsp+60h] [rbp-170h]
  int v45; // [rsp+78h] [rbp-158h] BYREF
  __int64 v46; // [rsp+80h] [rbp-150h]
  int *v47; // [rsp+88h] [rbp-148h]
  int *v48; // [rsp+90h] [rbp-140h]
  __int64 v49; // [rsp+98h] [rbp-138h]
  int v50; // [rsp+A8h] [rbp-128h] BYREF
  __int64 v51; // [rsp+B0h] [rbp-120h]
  int *v52; // [rsp+B8h] [rbp-118h]
  int *v53; // [rsp+C0h] [rbp-110h]
  __int64 v54; // [rsp+C8h] [rbp-108h]
  __int16 v55; // [rsp+D0h] [rbp-100h]
  char v56; // [rsp+D2h] [rbp-FEh]
  int v57; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+E8h] [rbp-E8h]
  int *v59; // [rsp+F0h] [rbp-E0h]
  int *v60; // [rsp+F8h] [rbp-D8h]
  __int64 v61; // [rsp+100h] [rbp-D0h]
  int v62; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+118h] [rbp-B8h]
  int *v64; // [rsp+120h] [rbp-B0h]
  int *v65; // [rsp+128h] [rbp-A8h]
  __int64 v66; // [rsp+130h] [rbp-A0h]
  __int64 v67; // [rsp+138h] [rbp-98h]
  __int64 v68; // [rsp+140h] [rbp-90h]
  void *v69; // [rsp+148h] [rbp-88h]
  __int64 v70; // [rsp+150h] [rbp-80h]
  _BYTE v71[32]; // [rsp+158h] [rbp-78h] BYREF
  void *src; // [rsp+178h] [rbp-58h]
  unsigned int v73; // [rsp+180h] [rbp-50h]
  int v74; // [rsp+184h] [rbp-4Ch]
  _QWORD v75[9]; // [rsp+188h] [rbp-48h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F99CCD )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_39;
  }
  v5 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_4F99CCD)
                 + 160);
  v33[0] = a1;
  v35 = sub_142B500;
  v34 = sub_142B5B0;
  sub_1432E70((__int64)v36, a2, (__int64)v33, v5);
  if ( *(_BYTE *)(a1 + 552) )
    sub_142D3E0(a1 + 160);
  v6 = v38;
  *(_BYTE *)(a1 + 552) = 1;
  v7 = a1 + 168;
  if ( v6 )
  {
    v8 = v37;
    *(_QWORD *)(a1 + 176) = v6;
    *(_DWORD *)(a1 + 168) = v8;
    *(_QWORD *)(a1 + 184) = v39;
    *(_QWORD *)(a1 + 192) = v40;
    *(_QWORD *)(v6 + 8) = v7;
    v38 = 0;
    *(_QWORD *)(a1 + 200) = v41;
    v39 = &v37;
    v40 = &v37;
    v41 = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 168) = 0;
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 184) = v7;
    *(_QWORD *)(a1 + 192) = v7;
    *(_QWORD *)(a1 + 200) = 0;
  }
  v9 = v42;
  v10 = a1 + 248;
  v42 = 0;
  *(_QWORD *)(a1 + 208) = v9;
  v11 = v43;
  v43 = 0;
  *(_QWORD *)(a1 + 216) = v11;
  v12 = v44;
  LODWORD(v44) = 0;
  *(_QWORD *)(a1 + 224) = v12;
  v13 = v46;
  if ( v46 )
  {
    v14 = v45;
    *(_QWORD *)(a1 + 256) = v46;
    *(_DWORD *)(a1 + 248) = v14;
    *(_QWORD *)(a1 + 264) = v47;
    *(_QWORD *)(a1 + 272) = v48;
    *(_QWORD *)(v13 + 8) = v10;
    v46 = 0;
    *(_QWORD *)(a1 + 280) = v49;
    v47 = &v45;
    v48 = &v45;
    v49 = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 264) = v10;
    *(_QWORD *)(a1 + 272) = v10;
    *(_QWORD *)(a1 + 280) = 0;
  }
  v15 = v51;
  v16 = a1 + 296;
  if ( v51 )
  {
    v17 = v50;
    *(_QWORD *)(a1 + 304) = v51;
    *(_DWORD *)(a1 + 296) = v17;
    *(_QWORD *)(a1 + 312) = v52;
    *(_QWORD *)(a1 + 320) = v53;
    *(_QWORD *)(v15 + 8) = v16;
    v51 = 0;
    *(_QWORD *)(a1 + 328) = v54;
    v52 = &v50;
    v53 = &v50;
    v54 = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 296) = 0;
    *(_QWORD *)(a1 + 304) = 0;
    *(_QWORD *)(a1 + 312) = v16;
    *(_QWORD *)(a1 + 320) = v16;
    *(_QWORD *)(a1 + 328) = 0;
  }
  v18 = a1 + 352;
  *(_WORD *)(a1 + 336) = v55;
  *(_BYTE *)(a1 + 338) = v56;
  v19 = v58;
  if ( v58 )
  {
    v20 = v57;
    *(_QWORD *)(a1 + 360) = v58;
    *(_DWORD *)(a1 + 352) = v20;
    *(_QWORD *)(a1 + 368) = v59;
    *(_QWORD *)(a1 + 376) = v60;
    *(_QWORD *)(v19 + 8) = v18;
    v58 = 0;
    *(_QWORD *)(a1 + 384) = v61;
    v59 = &v57;
    v60 = &v57;
    v61 = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 352) = 0;
    *(_QWORD *)(a1 + 360) = 0;
    *(_QWORD *)(a1 + 368) = v18;
    *(_QWORD *)(a1 + 376) = v18;
    *(_QWORD *)(a1 + 384) = 0;
  }
  v21 = v63;
  v22 = a1 + 400;
  if ( v63 )
  {
    v23 = v62;
    *(_QWORD *)(a1 + 408) = v63;
    *(_DWORD *)(a1 + 400) = v23;
    *(_QWORD *)(a1 + 416) = v64;
    *(_QWORD *)(a1 + 424) = v65;
    *(_QWORD *)(v21 + 8) = v22;
    v63 = 0;
    *(_QWORD *)(a1 + 432) = v66;
    v64 = &v62;
    v65 = &v62;
    v66 = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 400) = 0;
    *(_QWORD *)(a1 + 408) = 0;
    *(_QWORD *)(a1 + 416) = v22;
    *(_QWORD *)(a1 + 424) = v22;
    *(_QWORD *)(a1 + 432) = 0;
  }
  v24 = v67;
  v25 = v70;
  *(_QWORD *)(a1 + 464) = 0x400000000LL;
  *(_QWORD *)(a1 + 440) = v24;
  *(_QWORD *)(a1 + 448) = v68;
  v26 = (void *)(a1 + 472);
  *(_QWORD *)(a1 + 456) = a1 + 472;
  if ( v25 )
  {
    v30 = v69;
    if ( v69 == v71 )
    {
      v32 = 8LL * v25;
      if ( v25 <= 4
        || (sub_16CD150(a1 + 456, a1 + 472, v25, 8),
            v26 = *(void **)(a1 + 456),
            v30 = v69,
            (v32 = 8LL * (unsigned int)v70) != 0) )
      {
        memcpy(v26, v30, v32);
      }
      *(_DWORD *)(a1 + 464) = v25;
      LODWORD(v70) = 0;
    }
    else
    {
      v31 = HIDWORD(v70);
      *(_QWORD *)(a1 + 456) = v69;
      *(_DWORD *)(a1 + 464) = v25;
      *(_DWORD *)(a1 + 468) = v31;
      v69 = v71;
      v70 = 0;
    }
  }
  v27 = v73;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 504) = a1 + 520;
  if ( v27 )
  {
    if ( src == v75 )
    {
      sub_16CD150(a1 + 504, a1 + 520, v27, 16);
      if ( 16LL * v73 )
        memcpy(*(void **)(a1 + 504), src, 16LL * v73);
      *(_DWORD *)(a1 + 512) = v27;
    }
    else
    {
      *(_QWORD *)(a1 + 504) = src;
      v29 = v74;
      *(_DWORD *)(a1 + 512) = v27;
      *(_DWORD *)(a1 + 516) = v29;
      src = v75;
      v74 = 0;
    }
  }
  v68 = 0;
  v67 = 0;
  *(_QWORD *)(a1 + 520) = v75[0];
  v75[0] = 0;
  *(_QWORD *)(a1 + 528) = v75[1];
  LODWORD(v70) = 0;
  *(_QWORD *)(a1 + 544) = v75[3];
  v73 = 0;
  sub_142D3E0((__int64)v36);
  if ( v34 )
    v34(v33, v33, 3);
  return 0;
}
