// Function: sub_339D300
// Address: 0x339d300
//
__int64 __fastcall sub_339D300(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _DWORD *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  __int64 result; // rax
  int v15; // edx
  __int64 v16; // rax
  int v17; // r14d
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  _DWORD *v19; // rax
  char v20; // cl
  int v21; // edx
  unsigned __int16 v22; // ax
  __int64 *v23; // rdi
  unsigned int v24; // r12d
  int v25; // eax
  __int64 v26; // r9
  int v27; // r8d
  int v28; // ecx
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  int v34; // edx
  __int64 (__fastcall *v35)(__int64, __int64, unsigned int); // rax
  unsigned __int16 v36; // r14
  int v37; // eax
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rsi
  int v43; // edx
  int v44; // edx
  char v45; // r12
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // r9
  char v50; // al
  int v51; // edx
  int v52; // edx
  unsigned __int16 v53; // r14
  __int64 v54; // rax
  int v55; // edx
  char v56; // [rsp+8h] [rbp-E8h]
  __int64 v57; // [rsp+8h] [rbp-E8h]
  int v58; // [rsp+10h] [rbp-E0h]
  __int64 v59; // [rsp+10h] [rbp-E0h]
  __int64 v60; // [rsp+10h] [rbp-E0h]
  int v61; // [rsp+18h] [rbp-D8h]
  __int64 v62; // [rsp+18h] [rbp-D8h]
  __int64 v63; // [rsp+18h] [rbp-D8h]
  __int64 v64; // [rsp+18h] [rbp-D8h]
  __int16 v65; // [rsp+1Ah] [rbp-D6h]
  __int64 v69; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v70; // [rsp+A0h] [rbp-50h] BYREF
  char v71; // [rsp+A8h] [rbp-48h]
  __int64 v72; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-38h]

  v10 = *(_QWORD *)(a6 + 864);
  v11 = *(_QWORD *)(v10 + 16);
  v69 = sub_2E79000(*(__int64 **)(v10 + 40));
  if ( *(_BYTE *)a1 > 0x15u )
  {
    if ( *(_BYTE *)a1 != 63 )
      return 0;
    if ( a7 != *(_QWORD *)(a1 + 40) )
      return 0;
    if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
      return 0;
    v62 = *(_QWORD *)(a1 - 64);
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v62 + 8) + 8LL) - 17 <= 1 )
      return 0;
    v59 = *(_QWORD *)(a1 - 32);
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v59 + 8) + 8LL) - 17 > 1 )
      return 0;
    v57 = *(_QWORD *)(a1 + 80);
    v45 = sub_AE5020(v69, v57);
    v72 = sub_9208B0(v69, v57);
    v73 = v46;
    v71 = v46;
    v70 = ((1LL << v45) + ((unsigned __int64)(v72 + 7) >> 3) - 1) >> v45 << v45;
    if ( (_BYTE)v46 )
      return 0;
    v47 = sub_CA1930(&v70);
    v48 = v59;
    v49 = v62;
    if ( v47 != 1 )
    {
      v60 = v62;
      v63 = v48;
      v50 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v11 + 712LL))(v11, v70, a8);
      v48 = v63;
      v49 = v60;
      if ( !v50 )
        return 0;
    }
    v64 = v48;
    *(_QWORD *)a2 = sub_338B750(a6, v49);
    *(_DWORD *)(a2 + 8) = v51;
    *(_QWORD *)a3 = sub_338B750(a6, v64);
    *(_DWORD *)(a3 + 8) = v52;
    *a4 = 0;
    v53 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v11 + 32LL))(v11, v69, 0);
    sub_336E8F0((__int64)&v72, *(_QWORD *)a6, *(_DWORD *)(a6 + 848));
    v54 = sub_CA1930(&v70);
    *(_QWORD *)a5 = sub_3400BD0(v10, v54, (unsigned int)&v72, v53, 0, 1, 0, v54);
    *(_DWORD *)(a5 + 8) = v55;
    sub_9C6650(&v72);
    return 1;
  }
  v13 = sub_AD7630(a1, 0, v12);
  result = 0;
  if ( !v13 )
    return result;
  *(_QWORD *)a2 = sub_338B750(a6, (__int64)v13);
  *(_DWORD *)(a2 + 8) = v15;
  v16 = *(_QWORD *)(a1 + 8);
  v17 = *(_DWORD *)(v16 + 32);
  v56 = *(_BYTE *)(v16 + 8);
  v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v11 + 32LL);
  if ( v18 == sub_2D42F30 )
  {
    v19 = sub_AE2980(v69, 0);
    v20 = v56;
    v21 = v19[1];
    v22 = 2;
    if ( v21 != 1 )
    {
      v22 = 3;
      if ( v21 != 2 )
      {
        v22 = 4;
        if ( v21 != 4 )
        {
          v22 = 5;
          if ( v21 != 8 )
          {
            v22 = 6;
            if ( v21 != 16 )
            {
              v22 = 7;
              if ( v21 != 32 )
              {
                v22 = 8;
                if ( v21 != 64 )
                  v22 = 9 * (v21 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v22 = v18(v11, v69, 0);
    v20 = v56;
  }
  v23 = *(__int64 **)(v10 + 64);
  LODWORD(v72) = v17;
  v24 = v22;
  BYTE4(v72) = v56 == 18;
  if ( v20 == 18 )
  {
    LOWORD(v25) = sub_2D43AD0(v22, v17);
    v27 = 0;
    if ( (_WORD)v25 )
      goto LABEL_14;
  }
  else
  {
    LOWORD(v25) = sub_2D43050(v22, v17);
    v27 = 0;
    if ( (_WORD)v25 )
      goto LABEL_14;
  }
  v25 = sub_3009450(v23, v24, 0, v72, 0, v26);
  v65 = HIWORD(v25);
  v27 = v44;
LABEL_14:
  HIWORD(v28) = v65;
  v29 = *(_DWORD *)(a6 + 848);
  v72 = 0;
  LOWORD(v28) = v25;
  v30 = *(_QWORD *)a6;
  LODWORD(v73) = v29;
  v61 = v28;
  if ( v30 )
  {
    if ( &v72 != (__int64 *)(v30 + 48) )
    {
      v31 = *(_QWORD *)(v30 + 48);
      v72 = v31;
      if ( v31 )
      {
        v58 = v27;
        sub_B96E90((__int64)&v72, v31, 1);
        v27 = v58;
      }
    }
  }
  v32 = sub_3400BD0(v10, 0, (unsigned int)&v72, v61, v27, 0, 0);
  v33 = v72;
  *(_QWORD *)a3 = v32;
  *(_DWORD *)(a3 + 8) = v34;
  if ( v33 )
    sub_B91220((__int64)&v72, v33);
  *a4 = 0;
  v35 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v11 + 32LL);
  if ( v35 == sub_2D42F30 )
  {
    v36 = 2;
    v37 = sub_AE2980(v69, 0)[1];
    if ( v37 != 1 )
    {
      v36 = 3;
      if ( v37 != 2 )
      {
        v36 = 4;
        if ( v37 != 4 )
        {
          v36 = 5;
          if ( v37 != 8 )
          {
            v36 = 6;
            if ( v37 != 16 )
            {
              v36 = 7;
              if ( v37 != 32 )
              {
                v36 = 8;
                if ( v37 != 64 )
                  v36 = 9 * (v37 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v36 = v35(v11, v69, 0);
  }
  v38 = *(_DWORD *)(a6 + 848);
  v39 = *(_QWORD *)a6;
  v72 = 0;
  LODWORD(v73) = v38;
  if ( v39 )
  {
    if ( &v72 != (__int64 *)(v39 + 48) )
    {
      v40 = *(_QWORD *)(v39 + 48);
      v72 = v40;
      if ( v40 )
        sub_B96E90((__int64)&v72, v40, 1);
    }
  }
  v41 = sub_3400BD0(v10, 1, (unsigned int)&v72, v36, 0, 1, 0);
  v42 = v72;
  *(_QWORD *)a5 = v41;
  *(_DWORD *)(a5 + 8) = v43;
  if ( v42 )
    sub_B91220((__int64)&v72, v42);
  return 1;
}
