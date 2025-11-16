// Function: sub_1C16A20
// Address: 0x1c16a20
//
__int64 __fastcall sub_1C16A20(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  char v5; // al
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  char v9; // al
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  char v13; // al
  char v14; // al
  _BOOL8 v15; // rcx
  char v16; // al
  char v17; // al
  char v18; // al
  _BOOL8 v19; // rcx
  char v20; // al
  char v21; // al
  char v22; // al
  _BOOL8 v23; // rcx
  char v24; // al
  char v25; // al
  char v26; // al
  _BOOL8 v27; // rcx
  char v28; // al
  unsigned __int8 v29; // al
  char v30; // al
  _BOOL8 v31; // rcx
  char v32; // al
  char v33; // al
  _BOOL8 v34; // rcx
  char v35; // al
  char v36; // al
  char v37; // al
  _BOOL8 v38; // rcx
  char v39; // al
  char v40; // al
  char v41; // al
  _BOOL8 v42; // rcx
  char v43; // al
  char v44; // al
  char v45; // al
  _BOOL8 v46; // rcx
  char v47; // al
  char v48; // al
  char v49; // al
  _BOOL8 v50; // rcx
  char v51; // al
  char v52; // al
  _BOOL8 v53; // rcx
  unsigned int v54; // eax
  char v55; // al
  _BOOL8 v56; // rcx
  __int64 result; // rax
  unsigned __int8 (__fastcall *v58)(__int64, const char *, _BOOL8); // r13
  char v59; // al
  _BOOL8 v60; // rdx
  unsigned __int8 (__fastcall *v61)(__int64, const char *, _BOOL8); // r13
  char v62; // al
  _BOOL8 v63; // rdx
  unsigned __int8 (__fastcall *v64)(__int64, const char *, _BOOL8); // r13
  char v65; // al
  _BOOL8 v66; // rdx
  unsigned __int8 (__fastcall *v67)(__int64, const char *, _BOOL8); // r13
  char v68; // al
  _BOOL8 v69; // rdx
  char v70; // [rsp+3h] [rbp-3Dh] BYREF
  int v71; // [rsp+4h] [rbp-3Ch] BYREF
  _QWORD v72[7]; // [rsp+8h] [rbp-38h] BYREF

  v71 = *(_BYTE *)a2 & 1;
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = v71 == 0;
  v4 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IgnoreInf",
         0,
         v3,
         &v70,
         v72);
  if ( v4 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v4 = v71 & 1;
  }
  else if ( !v70 )
  {
    v4 = v71 & 1;
  }
  v5 = v4 & 1 | *(_BYTE *)a2 & 0xFE;
  *(_BYTE *)a2 = v5;
  v71 = (v5 & 2) != 0;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = v71 == 0;
  v8 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IgnoreNaN",
         0,
         v7,
         &v70,
         v72);
  if ( v8 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v8 = v71 & 1;
  }
  else if ( !v70 )
  {
    v8 = v71 & 1;
  }
  v9 = (2 * (v8 & 1)) | *(_BYTE *)a2 & 0xFD;
  *(_BYTE *)a2 = v9;
  v71 = (v9 & 4) != 0;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = v71 == 0;
  v12 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "IgnoreSignedZero",
          0,
          v11,
          &v70,
          v72);
  if ( v12 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v12 = v71 & 1;
  }
  else if ( !v70 )
  {
    v12 = v71 & 1;
  }
  v13 = (4 * (v12 & 1)) | *(_BYTE *)a2 & 0xFB;
  *(_BYTE *)a2 = v13;
  v71 = (v13 & 8) != 0;
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v15 = 0;
  if ( v14 )
    v15 = v71 == 0;
  v16 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "ReorderFloat",
          0,
          v15,
          &v70,
          v72);
  if ( v16 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v16 = v71 & 1;
  }
  else if ( !v70 )
  {
    v16 = v71 & 1;
  }
  v17 = (8 * (v16 & 1)) | *(_BYTE *)a2 & 0xF7;
  *(_BYTE *)a2 = v17;
  v71 = (v17 & 0x10) != 0;
  v18 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v19 = 0;
  if ( v18 )
    v19 = v71 == 0;
  v20 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "ReorderHalf",
          0,
          v19,
          &v70,
          v72);
  if ( v20 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v20 = v71 & 1;
  }
  else if ( !v70 )
  {
    v20 = v71 & 1;
  }
  v21 = (16 * (v20 & 1)) | *(_BYTE *)a2 & 0xEF;
  *(_BYTE *)a2 = v21;
  v71 = (v21 & 0x20) != 0;
  v22 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v23 = 0;
  if ( v22 )
    v23 = v71 == 0;
  v24 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "Ftz",
          0,
          v23,
          &v70,
          v72);
  if ( v24 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v24 = v71 & 1;
  }
  else if ( !v70 )
  {
    v24 = v71 & 1;
  }
  v25 = (32 * (v24 & 1)) | *(_BYTE *)a2 & 0xDF;
  *(_BYTE *)a2 = v25;
  v71 = (v25 & 0x40) != 0;
  v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v27 = 0;
  if ( v26 )
    v27 = v71 == 0;
  v28 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "FastSqrt",
          0,
          v27,
          &v70,
          v72);
  if ( v28 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v28 = v71 & 1;
  }
  else if ( !v70 )
  {
    v28 = v71 & 1;
  }
  v29 = ((v28 & 1) << 6) | *(_BYTE *)a2 & 0xBF;
  *(_BYTE *)a2 = v29;
  v71 = v29 >> 7;
  v30 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v31 = 0;
  if ( v30 )
    v31 = v71 == 0;
  v32 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "Fmad",
          0,
          v31,
          &v70,
          v72);
  if ( v32 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v32 = v71 & 1;
  }
  else if ( !v70 )
  {
    v32 = v71 & 1;
  }
  *(_BYTE *)a2 = *(_BYTE *)a2 & 0x7F | (v32 << 7);
  v71 = *(_BYTE *)(a2 + 1) & 1;
  v33 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v34 = 0;
  if ( v33 )
    v34 = v71 == 0;
  v35 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "AllowRcpRsqToSqrt",
          0,
          v34,
          &v70,
          v72);
  if ( v35 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v35 = v71 & 1;
  }
  else if ( !v70 )
  {
    v35 = v71 & 1;
  }
  v36 = v35 & 1 | *(_BYTE *)(a2 + 1) & 0xFE;
  *(_BYTE *)(a2 + 1) = v36;
  v71 = (v36 & 2) != 0;
  v37 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v38 = 0;
  if ( v37 )
    v38 = v71 == 0;
  v39 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "CanReorderFloatDistribute",
          0,
          v38,
          &v70,
          v72);
  if ( v39 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v39 = v71 & 1;
  }
  else if ( !v70 )
  {
    v39 = v71 & 1;
  }
  v40 = (2 * (v39 & 1)) | *(_BYTE *)(a2 + 1) & 0xFD;
  *(_BYTE *)(a2 + 1) = v40;
  v71 = (v40 & 4) != 0;
  v41 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v42 = 0;
  if ( v41 )
    v42 = v71 == 0;
  v43 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "ReassociateFloatAddOverMad",
          0,
          v42,
          &v70,
          v72);
  if ( v43 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v43 = v71 & 1;
  }
  else if ( !v70 )
  {
    v43 = v71 & 1;
  }
  v44 = (4 * (v43 & 1)) | *(_BYTE *)(a2 + 1) & 0xFB;
  *(_BYTE *)(a2 + 1) = v44;
  v71 = (v44 & 8) != 0;
  v45 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v46 = 0;
  if ( v45 )
    v46 = v71 == 0;
  v47 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "NoFloatMAD",
          0,
          v46,
          &v70,
          v72);
  if ( v47 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v47 = v71 & 1;
  }
  else if ( !v70 )
  {
    v47 = v71 & 1;
  }
  v48 = (8 * (v47 & 1)) | *(_BYTE *)(a2 + 1) & 0xF7;
  *(_BYTE *)(a2 + 1) = v48;
  v71 = (v48 & 0x10) != 0;
  v49 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v50 = 0;
  if ( v49 )
    v50 = v71 == 0;
  v51 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "LaxFP16ApproximateDivision",
          0,
          v50,
          &v70,
          v72);
  if ( v51 )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v51 = v71 & 1;
  }
  else if ( !v70 )
  {
    v51 = v71 & 1;
  }
  *(_BYTE *)(a2 + 1) = *(_BYTE *)(a2 + 1) & 0xEF | (16 * (v51 & 1));
  v71 = *(_DWORD *)a2 >> 13;
  v52 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v53 = 0;
  if ( v52 )
    v53 = v71 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Reserved",
         0,
         v53,
         &v70,
         v72) )
  {
    sub_1C14710(a1, (unsigned int *)&v71);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
    v54 = v71 & 0x7FFFF;
  }
  else
  {
    v54 = 0;
    if ( !v70 )
      v54 = v71 & 0x7FFFF;
  }
  *(_DWORD *)a2 = (v54 << 13) | *(_DWORD *)a2 & 0x1FFF;
  v55 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v56 = 0;
  if ( v55 )
    v56 = *(_DWORD *)(a2 + 4) == 0;
  result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, int *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "Divide",
             0,
             v56,
             &v71,
             v72);
  if ( (_BYTE)result )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v58 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v59 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v60 = 0;
    if ( v59 )
      v60 = *(_DWORD *)(a2 + 4) == 0;
    if ( v58(a1, "NVVM_FAST_MATH_DIVIDE_PRECISE_NO_FTZ", v60) )
      *(_DWORD *)(a2 + 4) = 0;
    v61 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v62 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v63 = 0;
    if ( v62 )
      v63 = *(_DWORD *)(a2 + 4) == 1;
    if ( v61(a1, "NVVM_FAST_MATH_DIVIDE_PRECISE_ALLOW_FTZ", v63) )
      *(_DWORD *)(a2 + 4) = 1;
    v64 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v65 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v66 = 0;
    if ( v65 )
      v66 = *(_DWORD *)(a2 + 4) == 2;
    if ( v64(a1, "NVVM_FAST_MATH_DIVIDE_FULL_RANGE_APPROX", v66) )
      *(_DWORD *)(a2 + 4) = 2;
    v67 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v68 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v69 = 0;
    if ( v68 )
      v69 = *(_DWORD *)(a2 + 4) == 3;
    if ( v67(a1, "NVVM_FAST_MATH_DIVIDE_FAST_APPROX", v69) )
      *(_DWORD *)(a2 + 4) = 3;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v72[0]);
  }
  else if ( (_BYTE)v71 )
  {
    *(_DWORD *)(a2 + 4) = 0;
  }
  return result;
}
