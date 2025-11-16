// Function: sub_33F3590
// Address: 0x33f3590
//
_QWORD *__fastcall sub_33F3590(
        _QWORD *a1,
        char a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  __m128i *v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // r14
  int v15; // edx
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  int v19; // edx
  unsigned __int16 v20; // ax
  _QWORD *v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // r9
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  _QWORD *v33; // r12
  unsigned __int64 v35; // rbx
  int v36; // r8d
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rcx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  int v43; // [rsp+0h] [rbp-110h]
  unsigned __int64 v44; // [rsp+8h] [rbp-108h]
  int v45; // [rsp+10h] [rbp-100h]
  int v46; // [rsp+14h] [rbp-FCh]
  int v48; // [rsp+18h] [rbp-F8h]
  __int64 *v49; // [rsp+20h] [rbp-F0h] BYREF
  unsigned __int8 *v50; // [rsp+28h] [rbp-E8h] BYREF
  unsigned __int64 v51[4]; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE *v52; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-B8h]
  _BYTE v54[176]; // [rsp+60h] [rbp-B0h] BYREF

  v46 = (a2 == 0) + 366;
  v12 = sub_33ED250((__int64)a1, 1, 0);
  v13 = (__int64 *)a1[5];
  v51[1] = a5;
  v14 = a1[2];
  v44 = (unsigned __int64)v12;
  v43 = v15;
  v51[0] = a4;
  v16 = sub_2E79000(v13);
  v17 = *(_DWORD *)(v16 + 4);
  v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v14 + 32LL);
  if ( v18 == sub_2D42F30 )
  {
    v19 = sub_AE2980(v16, v17)[1];
    v20 = 2;
    if ( v19 != 1 )
    {
      v20 = 3;
      if ( v19 != 2 )
      {
        v20 = 4;
        if ( v19 != 4 )
        {
          v20 = 5;
          if ( v19 != 8 )
          {
            v20 = 6;
            if ( v19 != 16 )
            {
              v20 = 7;
              if ( v19 != 32 )
              {
                v20 = 8;
                if ( v19 != 64 )
                  v20 = 9 * (v19 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v20 = v18(v14, v16, v17);
  }
  v21 = sub_33EDBD0(a1, a6, v20, 0, 1);
  v51[3] = v22;
  v51[2] = (unsigned __int64)v21;
  v53 = 0x2000000000LL;
  v52 = v54;
  sub_33C9670((__int64)&v52, v46, v44, v51, 2, v23);
  v26 = (unsigned int)v53;
  v27 = (unsigned int)v53 + 1LL;
  if ( v27 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, v27, 4u, v24, v25);
    v26 = (unsigned int)v53;
  }
  *(_DWORD *)&v52[4 * v26] = a6;
  LODWORD(v53) = v53 + 1;
  v28 = (unsigned int)v53;
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v24, v25);
    v28 = (unsigned int)v53;
  }
  *(_DWORD *)&v52[4 * v28] = a7;
  LODWORD(v53) = v53 + 1;
  v29 = (unsigned int)v53;
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v24, v25);
    v29 = (unsigned int)v53;
  }
  *(_DWORD *)&v52[4 * v29] = HIDWORD(a7);
  LODWORD(v53) = v53 + 1;
  v30 = (unsigned int)v53;
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v24, v25);
    v30 = (unsigned int)v53;
  }
  *(_DWORD *)&v52[4 * v30] = a8;
  LODWORD(v53) = v53 + 1;
  v31 = (unsigned int)v53;
  if ( (unsigned __int64)(unsigned int)v53 + 1 > HIDWORD(v53) )
  {
    sub_C8D5F0((__int64)&v52, v54, (unsigned int)v53 + 1LL, 4u, v24, v25);
    v31 = (unsigned int)v53;
  }
  *(_DWORD *)&v52[4 * v31] = HIDWORD(a8);
  LODWORD(v53) = v53 + 1;
  v49 = 0;
  v32 = sub_33CCCF0((__int64)a1, (__int64)&v52, a3, (__int64 *)&v49);
  if ( v32 )
  {
    v33 = v32;
    goto LABEL_22;
  }
  v35 = a1[52];
  v36 = *(_DWORD *)(a3 + 8);
  if ( v35 )
  {
    a1[52] = *(_QWORD *)v35;
LABEL_28:
    v37 = *(_QWORD *)a3;
    v50 = (unsigned __int8 *)v37;
    if ( v37 )
    {
      v48 = v36;
      sub_B96E90((__int64)&v50, v37, 1);
      v36 = v48;
    }
    *(_QWORD *)v35 = 0;
    v38 = v50;
    *(_QWORD *)(v35 + 8) = 0;
    *(_DWORD *)(v35 + 24) = v46;
    *(_QWORD *)(v35 + 16) = 0;
    *(_QWORD *)(v35 + 48) = v44;
    *(_DWORD *)(v35 + 28) = 0;
    *(_WORD *)(v35 + 34) = -1;
    *(_DWORD *)(v35 + 36) = -1;
    *(_QWORD *)(v35 + 40) = 0;
    *(_QWORD *)(v35 + 56) = 0;
    *(_DWORD *)(v35 + 64) = 0;
    *(_DWORD *)(v35 + 68) = v43;
    *(_DWORD *)(v35 + 72) = v36;
    *(_QWORD *)(v35 + 80) = v38;
    if ( v38 )
      sub_B976B0((__int64)&v50, v38, v35 + 80);
    *(_QWORD *)(v35 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v35 + 32) = 0;
    *(_QWORD *)(v35 + 96) = a7;
    *(_QWORD *)(v35 + 104) = a8;
    goto LABEL_33;
  }
  v40 = a1[53];
  a1[63] += 120LL;
  v41 = (v40 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v41 + 120 || !v40 )
  {
    v45 = v36;
    v42 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    v36 = v45;
    v35 = v42;
    goto LABEL_28;
  }
  a1[53] = v41 + 120;
  if ( v41 )
  {
    v35 = (v40 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_28;
  }
LABEL_33:
  sub_33E4EC0((__int64)a1, v35, (__int64)v51, 2);
  sub_C657C0(a1 + 65, (__int64 *)v35, v49, (__int64)off_4A367D0);
  v39 = (__int64)a1;
  v33 = (_QWORD *)v35;
  sub_33CC420(v39, v35);
LABEL_22:
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return v33;
}
