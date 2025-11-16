// Function: sub_37846E0
// Address: 0x37846e0
//
void __fastcall sub_37846E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r12
  unsigned int *v7; // rax
  unsigned int v10; // ebx
  unsigned __int64 v11; // rdx
  unsigned int *i; // rdx
  __int64 v13; // r12
  __int64 v14; // rcx
  unsigned int *v15; // rdx
  __int64 v16; // rdx
  unsigned int *v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int16 v21; // r15
  __int64 v22; // r10
  unsigned __int16 *v23; // rax
  __int64 v24; // rsi
  _QWORD *v25; // rdi
  unsigned __int8 *v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rdx
  unsigned __int8 *v29; // rax
  __int64 v30; // r9
  __int64 v31; // rdx
  unsigned __int64 v32; // r13
  unsigned __int64 v33; // r8
  unsigned int v34; // r14d
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // r8
  __int64 v37; // rdx
  unsigned __int16 *v38; // rax
  unsigned __int16 *v39; // rdx
  __int128 v40; // [rsp-20h] [rbp-1E0h]
  __int128 v41; // [rsp-10h] [rbp-1D0h]
  unsigned __int64 v42; // [rsp-10h] [rbp-1D0h]
  unsigned __int64 v43; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v44; // [rsp+28h] [rbp-198h]
  unsigned __int64 v45; // [rsp+28h] [rbp-198h]
  __int64 v46; // [rsp+30h] [rbp-190h]
  __int64 v47; // [rsp+30h] [rbp-190h]
  __int64 v48; // [rsp+38h] [rbp-188h]
  __int64 v49; // [rsp+38h] [rbp-188h]
  unsigned __int64 v50; // [rsp+38h] [rbp-188h]
  __int64 v51; // [rsp+40h] [rbp-180h] BYREF
  int v52; // [rsp+48h] [rbp-178h]
  unsigned __int8 *v53; // [rsp+50h] [rbp-170h] BYREF
  __int64 v54; // [rsp+58h] [rbp-168h]
  unsigned __int8 *v55; // [rsp+60h] [rbp-160h]
  __int64 v56; // [rsp+68h] [rbp-158h]
  unsigned int *v57; // [rsp+70h] [rbp-150h] BYREF
  __int64 v58; // [rsp+78h] [rbp-148h]
  _OWORD v59[8]; // [rsp+80h] [rbp-140h] BYREF
  unsigned __int16 *v60; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+108h] [rbp-B8h]
  _BYTE v62[176]; // [rsp+110h] [rbp-B0h] BYREF

  v7 = (unsigned int *)v59;
  v10 = *(_DWORD *)(a2 + 64);
  v57 = (unsigned int *)v59;
  v11 = 2 * v10;
  v58 = 0x800000000LL;
  if ( 2 * v10 )
  {
    if ( v11 > 8 )
    {
      sub_C8D5F0((__int64)&v57, v59, v11, 0x10u, a5, a6);
      v7 = &v57[4 * (unsigned int)v58];
      for ( i = &v57[8 * v10]; i != v7; v7 += 4 )
      {
LABEL_4:
        if ( v7 )
        {
          *(_QWORD *)v7 = 0;
          v7[2] = 0;
        }
      }
    }
    else
    {
      i = (unsigned int *)&v59[v11];
      if ( i != (unsigned int *)v59 )
        goto LABEL_4;
    }
    LODWORD(v58) = 2 * v10;
  }
  if ( v10 )
  {
    v44 = v6;
    v13 = 0;
    do
    {
      v14 = *(_QWORD *)(a2 + 40);
      v53 = 0;
      LODWORD(v54) = 0;
      v60 = 0;
      LODWORD(v61) = 0;
      sub_375E8D0(a1, *(_QWORD *)(v14 + 40 * v13), *(_QWORD *)(v14 + 40 * v13 + 8), (__int64)&v53, (__int64)&v60);
      v15 = &v57[4 * v13];
      *(_QWORD *)v15 = v53;
      v15[2] = v54;
      v16 = v10 + (unsigned int)v13++;
      v17 = &v57[4 * v16];
      *(_QWORD *)v17 = v60;
      v17[2] = v61;
    }
    while ( v10 != v13 );
    v18 = v10;
    v6 = v44;
    v19 = *(_QWORD *)(*(_QWORD *)v57 + 48LL) + 16LL * v57[2];
    v20 = v10;
    v21 = *(_WORD *)v19;
    v22 = *(_QWORD *)(v19 + 8);
    v23 = (unsigned __int16 *)v62;
    v61 = 0x800000000LL;
    v60 = (unsigned __int16 *)v62;
    if ( v10 > 8 )
    {
      v47 = v22;
      sub_C8D5F0((__int64)&v60, v62, v10, 0x10u, (__int64)&v60, v10);
      v18 = v10;
      v38 = v60;
      v39 = &v60[8 * v10];
      do
      {
        if ( v38 )
        {
          *v38 = v21;
          *((_QWORD *)v38 + 1) = v47;
        }
        v38 += 8;
      }
      while ( v38 != v39 );
    }
    else
    {
      do
      {
        *v23 = v21;
        v23 += 8;
        *((_QWORD *)v23 - 1) = v22;
        --v20;
      }
      while ( v20 );
    }
  }
  else
  {
    v18 = 0;
    HIDWORD(v61) = 8;
    v60 = (unsigned __int16 *)v62;
  }
  v24 = *(_QWORD *)(a2 + 80);
  LODWORD(v61) = v10;
  v51 = v24;
  if ( v24 )
  {
    v48 = v18;
    sub_B96E90((__int64)&v51, v24, 1);
    v18 = v48;
  }
  v25 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v41 + 1) = v18;
  v49 = v18;
  v52 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v41 = v57;
  v26 = sub_3411BE0(v25, 0xA3u, (__int64)&v51, v60, (unsigned int)v61, v18, v41);
  v27 = *(_QWORD **)(a1 + 8);
  v54 = v28;
  *((_QWORD *)&v40 + 1) = v49;
  v53 = v26;
  *(_QWORD *)&v40 = &v57[4 * v49];
  v29 = sub_3411BE0(v27, 0xA3u, (__int64)&v51, v60, (unsigned int)v61, v49, v40);
  v30 = 0;
  v55 = v29;
  v56 = v31;
  if ( v10 )
  {
    v45 = a2;
    v32 = v43;
    v33 = v6;
    v46 = a1;
    v34 = 0;
    do
    {
      v32 = ((2 * v34 + 1) % v10) | v32 & 0xFFFFFFFF00000000LL;
      v42 = (unsigned __int64)(&v53)[2 * ((2 * v34 + 1) / v10)];
      v35 = (unsigned __int64)(&v53)[2 * (2 * v34 / v10)];
      v36 = (2 * v34 % v10) | v33 & 0xFFFFFFFF00000000LL;
      v37 = v34++;
      v50 = v36;
      sub_3760810(v46, v45, v37, v35, v36, v30, v42, v32);
      v33 = v50;
    }
    while ( v10 != v34 );
  }
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  if ( v60 != (unsigned __int16 *)v62 )
    _libc_free((unsigned __int64)v60);
  if ( v57 != (unsigned int *)v59 )
    _libc_free((unsigned __int64)v57);
}
