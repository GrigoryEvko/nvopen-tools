// Function: sub_336F0F0
// Address: 0x336f0f0
//
__int64 __fastcall sub_336F0F0(__int64 a1, int a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  __int64 *v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // r15
  __int64 v10; // rax
  int v11; // edx
  unsigned __int16 v12; // ax
  int v13; // edx
  __int64 result; // rax
  __int64 v15; // r15
  __int64 v16; // r13
  __int64 v17; // rsi
  unsigned int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r14
  __int64 (*v26)(); // rax
  char v27; // r9
  __int64 (*v28)(); // rax
  __int64 v29; // r15
  unsigned __int8 v30; // al
  unsigned int v31; // eax
  unsigned int v32; // esi
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int); // rax
  _DWORD *v34; // rax
  __int64 v35; // r10
  unsigned int v36; // r9d
  int v37; // edx
  unsigned __int16 v38; // ax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r15
  __int64 v42; // r14
  unsigned int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // r13
  __int64 v46; // r8
  __int64 v47; // r9
  __m128i v48; // xmm0
  char v49; // al
  __int64 v50; // rax
  char v51; // al
  char v52; // [rsp+Ch] [rbp-104h]
  unsigned int v53; // [rsp+Ch] [rbp-104h]
  unsigned int v54; // [rsp+Ch] [rbp-104h]
  unsigned __int8 v55; // [rsp+Ch] [rbp-104h]
  __int64 v56; // [rsp+10h] [rbp-100h]
  int v57; // [rsp+10h] [rbp-100h]
  int v58; // [rsp+18h] [rbp-F8h]
  __int64 v59; // [rsp+20h] [rbp-F0h]
  __int64 v60; // [rsp+30h] [rbp-E0h]
  __int64 v61; // [rsp+30h] [rbp-E0h]
  __int64 v62; // [rsp+30h] [rbp-E0h]
  __int64 v63; // [rsp+30h] [rbp-E0h]
  __int64 v64; // [rsp+30h] [rbp-E0h]
  __int64 v66; // [rsp+38h] [rbp-D8h]
  __m128i v69; // [rsp+80h] [rbp-90h] BYREF
  __int64 v70; // [rsp+90h] [rbp-80h]
  __int128 v71; // [rsp+A0h] [rbp-70h]
  __int64 v72; // [rsp+B0h] [rbp-60h]
  __int64 v73; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v74; // [rsp+C8h] [rbp-48h]
  __int64 v75; // [rsp+D0h] [rbp-40h]
  __int64 v76; // [rsp+D8h] [rbp-38h]

  v7 = *(_QWORD *)(a4 + 232);
  v8 = *(__int64 **)(a5 + 40);
  if ( (unsigned __int8)(*(_BYTE *)v7 - 16) <= 2u || *(_BYTE *)v7 == 11 )
  {
    v66 = *(_QWORD *)(a5 + 16);
    v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v66 + 32LL);
    v10 = sub_2E79000(v8);
    if ( v9 == sub_2D42F30 )
    {
      v11 = sub_AE2980(v10, 0)[1];
      v12 = 2;
      if ( v11 != 1 )
      {
        v12 = 3;
        if ( v11 != 2 )
        {
          v12 = 4;
          if ( v11 != 4 )
          {
            v12 = 5;
            if ( v11 != 8 )
            {
              v12 = 6;
              if ( v11 != 16 )
              {
                v12 = 7;
                if ( v11 != 32 )
                {
                  v12 = 8;
                  if ( v11 != 64 )
                    v12 = 9 * (v11 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v12 = v9(v66, v10, 0);
    }
    *(_QWORD *)(a4 + 248) = sub_33EE5B0(a5, v7, v12, 0, 0, 0, 0, 0);
    *(_DWORD *)(a4 + 256) = v13;
    return a1;
  }
  else
  {
    v15 = *(_QWORD *)(v7 + 8);
    v60 = *(_QWORD *)(a5 + 16);
    v56 = v15;
    v16 = sub_2E79000(v8);
    v17 = v15;
    v18 = sub_AE5020(v16, v15);
    v19 = sub_9208B0(v16, v17);
    v20 = v18;
    v21 = 0;
    v73 = v19;
    v22 = v60;
    v74 = v23;
    v23 = (unsigned __int8)v23;
    v24 = *(_QWORD *)(*(_QWORD *)(a5 + 40) + 16LL);
    v59 = *(_QWORD *)(a5 + 40);
    v25 = ((1LL << v18) + ((unsigned __int64)(v19 + 7) >> 3) - 1) >> v18 << v18;
    v26 = *(__int64 (**)())(*(_QWORD *)v24 + 136LL);
    if ( v26 != sub_2DD19D0 )
    {
      v55 = v23;
      v50 = ((__int64 (__fastcall *)(__int64))v26)(v24);
      v23 = v55;
      v22 = v60;
      v21 = v50;
    }
    v27 = 0;
    if ( (_BYTE)v23 )
    {
      v28 = *(__int64 (**)())(*(_QWORD *)v21 + 328LL);
      if ( v28 != sub_2FDBCD0 )
      {
        v64 = v22;
        v51 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v28)(
                v21,
                1LL << v18,
                v23,
                v20,
                v24,
                0);
        v22 = v64;
        v27 = v51;
      }
    }
    v61 = v22;
    v52 = v27;
    v29 = *(_QWORD *)(v59 + 48);
    v30 = sub_AE5260(v16, v56);
    v31 = sub_2E77BD0(v29, v25, v30, 0, 0, v52);
    v32 = *(_DWORD *)(v16 + 4);
    v53 = v31;
    v33 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v61 + 32LL);
    if ( v33 == sub_2D42F30 )
    {
      v34 = sub_AE2980(v16, v32);
      v35 = v61;
      v36 = v53;
      v37 = v34[1];
      v38 = 2;
      if ( v37 != 1 )
      {
        v38 = 3;
        if ( v37 != 2 )
        {
          v38 = 4;
          if ( v37 != 4 )
          {
            v38 = 5;
            if ( v37 != 8 )
            {
              v38 = 6;
              if ( v37 != 16 )
              {
                v38 = 7;
                if ( v37 != 32 )
                {
                  v38 = 8;
                  if ( v37 != 64 )
                    v38 = 9 * (v37 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v38 = v33(v61, v16, v32);
      v36 = v53;
      v35 = v61;
    }
    v54 = v36;
    v62 = v35;
    v39 = sub_33EDBD0(a5, v36, v38, 0, 0);
    v41 = v40;
    v42 = v39;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    v76 = 0;
    v43 = sub_336EEB0(v62, v16, v56, 0);
    v63 = v44;
    v45 = v43;
    sub_2EAC300((__int64)&v69, v59, v54, 0);
    v46 = *(_QWORD *)(a4 + 248);
    v47 = *(_QWORD *)(a4 + 256);
    v48 = _mm_loadu_si128(&v69);
    v72 = v70;
    v57 = v46;
    v58 = v47;
    v71 = (__int128)v48;
    v49 = sub_33CC4A0(a5, v45, v63);
    result = sub_33F5040(a5, a1, a2, a3, v57, v58, v42, v41, v71, v72, v45, v63, v49, 0, (__int64)&v73);
    *(_QWORD *)(a4 + 248) = v42;
    *(_DWORD *)(a4 + 256) = v41;
  }
  return result;
}
