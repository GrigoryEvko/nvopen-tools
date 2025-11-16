// Function: sub_B873F0
// Address: 0xb873f0
//
__int64 __fastcall sub_B873F0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 **v7; // rdx
  __int64 *v8; // r8
  __int64 v9; // r13
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rsi
  int v16; // r11d
  __int64 v17; // r8
  __int64 **v18; // rdx
  unsigned int v19; // edi
  _QWORD *v20; // rax
  __int64 *v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  int v25; // edi
  int v26; // esi
  int v27; // r9d
  int v28; // r9d
  int v29; // edi
  unsigned int v30; // eax
  __int64 *v31; // rcx
  int v32; // r9d
  int v33; // eax
  int v34; // r8d
  int v35; // r8d
  __int64 v36; // rcx
  __int64 **v37; // r9
  unsigned int v38; // r14d
  int v39; // r10d
  __int64 *v40; // rax
  int v41; // r11d
  __int64 **v42; // r10
  __int64 v43; // [rsp+18h] [rbp-1B8h]
  __int64 v44; // [rsp+18h] [rbp-1B8h]
  __int64 v45; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v46; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v47; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v48; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v49; // [rsp+20h] [rbp-1B0h]
  __int64 v50; // [rsp+20h] [rbp-1B0h]
  _QWORD *v51; // [rsp+58h] [rbp-178h] BYREF
  _QWORD v52[2]; // [rsp+60h] [rbp-170h] BYREF
  _DWORD v53[32]; // [rsp+70h] [rbp-160h] BYREF
  _BYTE *v54; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+F8h] [rbp-D8h]
  _BYTE v56[64]; // [rsp+100h] [rbp-D0h] BYREF
  _BYTE *v57; // [rsp+140h] [rbp-90h] BYREF
  __int64 v58; // [rsp+148h] [rbp-88h]
  _BYTE v59[16]; // [rsp+150h] [rbp-80h] BYREF
  _BYTE *v60; // [rsp+160h] [rbp-70h] BYREF
  __int64 v61; // [rsp+168h] [rbp-68h]
  _BYTE v62[16]; // [rsp+170h] [rbp-60h] BYREF
  _BYTE *v63; // [rsp+180h] [rbp-50h] BYREF
  __int64 v64; // [rsp+188h] [rbp-48h]
  _BYTE v65[64]; // [rsp+190h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 680);
  v5 = *(_QWORD *)(a1 + 664);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 **)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 **)(v5 + 16 * v4) )
        return (__int64)v7[1];
    }
    else
    {
      v11 = 1;
      while ( v8 != (__int64 *)-4096LL )
      {
        v32 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 **)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v11 = v32;
      }
    }
  }
  v65[0] = 0;
  v54 = v56;
  v55 = 0x800000000LL;
  v57 = v59;
  v58 = 0x200000000LL;
  v61 = 0x200000000LL;
  v63 = v65;
  v12 = *a2;
  v60 = v62;
  v64 = 0;
  (*(void (__fastcall **)(__int64 *, _BYTE **))(v12 + 88))(a2, &v54);
  v52[0] = v53;
  v51 = v52;
  v53[0] = v65[0];
  v52[1] = 0x2000000001LL;
  sub_B803F0(&v51, (__int64)&v54);
  sub_B803F0(&v51, (__int64)&v57);
  sub_B803F0(&v51, (__int64)&v60);
  sub_B803F0(&v51, (__int64)&v63);
  v51 = 0;
  v13 = sub_C65B40(a1 + 544, v52, &v51, off_49DAC90);
  v14 = a1 + 544;
  if ( v13 )
  {
    v9 = v13 + 8;
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 560);
    *(_QWORD *)(a1 + 640) += 176LL;
    v24 = (v23 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(a1 + 568) >= v24 + 176 && v23 )
    {
      *(_QWORD *)(a1 + 560) = v24 + 176;
    }
    else
    {
      v24 = sub_9D1E70(a1 + 560, 176, 176, 3);
      v14 = a1 + 544;
    }
    *(_QWORD *)v24 = 0;
    v9 = v24 + 8;
    *(_QWORD *)(v24 + 8) = v24 + 24;
    v25 = v55;
    *(_QWORD *)(v24 + 16) = 0x800000000LL;
    if ( v25 )
    {
      v46 = v24;
      v50 = v14;
      sub_B7EC20(v24 + 8, (__int64)&v54);
      v24 = v46;
      v14 = v50;
    }
    *(_QWORD *)(v24 + 88) = v24 + 104;
    v26 = v58;
    *(_QWORD *)(v24 + 96) = 0x200000000LL;
    if ( v26 )
    {
      v45 = v14;
      v49 = v24;
      sub_B7EC20(v24 + 88, (__int64)&v57);
      v14 = v45;
      v24 = v49;
    }
    *(_QWORD *)(v24 + 120) = v24 + 136;
    *(_QWORD *)(v24 + 128) = 0x200000000LL;
    if ( (_DWORD)v61 )
    {
      v44 = v14;
      v48 = v24;
      sub_B7EC20(v24 + 120, (__int64)&v60);
      v14 = v44;
      v24 = v48;
    }
    *(_QWORD *)(v24 + 160) = 0;
    *(_QWORD *)(v24 + 152) = v24 + 168;
    if ( (_DWORD)v64 )
    {
      v43 = v14;
      v47 = v24;
      sub_B7EC20(v24 + 152, (__int64)&v63);
      v14 = v43;
      v24 = v47;
    }
    *(_BYTE *)(v24 + 168) = v65[0];
    sub_C657C0(v14, v24, v51, off_49DAC90);
  }
  v15 = *(unsigned int *)(a1 + 680);
  if ( !(_DWORD)v15 )
  {
    ++*(_QWORD *)(a1 + 656);
    goto LABEL_36;
  }
  v16 = 1;
  v17 = *(_QWORD *)(a1 + 664);
  v18 = 0;
  v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (_QWORD *)(v17 + 16LL * v19);
  v21 = (__int64 *)*v20;
  if ( (__int64 *)*v20 != a2 )
  {
    while ( v21 != (__int64 *)-4096LL )
    {
      if ( v21 == (__int64 *)-8192LL && !v18 )
        v18 = (__int64 **)v20;
      v19 = (v15 - 1) & (v16 + v19);
      v20 = (_QWORD *)(v17 + 16LL * v19);
      v21 = (__int64 *)*v20;
      if ( (__int64 *)*v20 == a2 )
        goto LABEL_12;
      ++v16;
    }
    if ( !v18 )
      v18 = (__int64 **)v20;
    v33 = *(_DWORD *)(a1 + 672);
    ++*(_QWORD *)(a1 + 656);
    v29 = v33 + 1;
    if ( 4 * (v33 + 1) < (unsigned int)(3 * v15) )
    {
      if ( (int)v15 - *(_DWORD *)(a1 + 676) - v29 > (unsigned int)v15 >> 3 )
      {
LABEL_38:
        *(_DWORD *)(a1 + 672) = v29;
        if ( *v18 != (__int64 *)-4096LL )
          --*(_DWORD *)(a1 + 676);
        *v18 = a2;
        v22 = (__int64 *)(v18 + 1);
        v18[1] = 0;
        goto LABEL_13;
      }
      sub_B85710(a1 + 656, v15);
      v34 = *(_DWORD *)(a1 + 680);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a1 + 664);
        v15 = *(unsigned int *)(a1 + 672);
        v37 = 0;
        v38 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v39 = 1;
        v29 = v15 + 1;
        v18 = (__int64 **)(v36 + 16LL * v38);
        v40 = *v18;
        if ( *v18 != a2 )
        {
          while ( v40 != (__int64 *)-4096LL )
          {
            if ( !v37 && v40 == (__int64 *)-8192LL )
              v37 = v18;
            v15 = (unsigned int)(v39 + 1);
            v38 = v35 & (v39 + v38);
            v18 = (__int64 **)(v36 + 16LL * v38);
            v40 = *v18;
            if ( *v18 == a2 )
              goto LABEL_38;
            ++v39;
          }
          if ( v37 )
            v18 = v37;
        }
        goto LABEL_38;
      }
LABEL_75:
      ++*(_DWORD *)(a1 + 672);
      BUG();
    }
LABEL_36:
    sub_B85710(a1 + 656, 2 * v15);
    v27 = *(_DWORD *)(a1 + 680);
    if ( v27 )
    {
      v28 = v27 - 1;
      v15 = *(_QWORD *)(a1 + 664);
      v29 = *(_DWORD *)(a1 + 672) + 1;
      v30 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = (__int64 **)(v15 + 16LL * v30);
      v31 = *v18;
      if ( *v18 != a2 )
      {
        v41 = 1;
        v42 = 0;
        while ( v31 != (__int64 *)-4096LL )
        {
          if ( v31 == (__int64 *)-8192LL && !v42 )
            v42 = v18;
          v30 = v28 & (v41 + v30);
          v18 = (__int64 **)(v15 + 16LL * v30);
          v31 = *v18;
          if ( *v18 == a2 )
            goto LABEL_38;
          ++v41;
        }
        if ( v42 )
          v18 = v42;
      }
      goto LABEL_38;
    }
    goto LABEL_75;
  }
LABEL_12:
  v22 = v20 + 1;
LABEL_13:
  *v22 = v9;
  if ( (_DWORD *)v52[0] != v53 )
    _libc_free(v52[0], v15);
  if ( v63 != v65 )
    _libc_free(v63, v15);
  if ( v60 != v62 )
    _libc_free(v60, v15);
  if ( v57 != v59 )
    _libc_free(v57, v15);
  if ( v54 != v56 )
    _libc_free(v54, v15);
  return v9;
}
