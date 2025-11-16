// Function: sub_33ADA20
// Address: 0x33ada20
//
void __fastcall sub_33ADA20(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 *v10; // r14
  unsigned int v11; // edx
  __int64 v12; // rdx
  unsigned __int16 v13; // cx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  _OWORD *v19; // rax
  _OWORD *i; // rdx
  __int64 v21; // r12
  __int64 v22; // rax
  int v23; // edx
  int v24; // edi
  __int64 v25; // rdx
  __int64 v26; // rax
  _OWORD *v27; // rax
  unsigned __int16 v28; // r12
  __int64 v29; // r8
  unsigned __int16 *v30; // rax
  _BYTE *v31; // rsi
  __int64 v32; // r12
  int v33; // eax
  int v34; // edx
  int v35; // r9d
  __int64 v36; // r8
  __int64 v37; // r9
  _BYTE *v38; // rcx
  _BYTE *v39; // rax
  _BYTE *v40; // rdx
  _BYTE *v41; // rcx
  __int64 v42; // rax
  _BYTE *v43; // rdx
  __int64 v44; // rax
  int v45; // edx
  int v46; // r12d
  _QWORD *v47; // rax
  unsigned __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rax
  _BYTE *v51; // rdx
  int v52; // r14d
  int v53; // eax
  __int64 v54; // r12
  int v55; // edx
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r12
  int v60; // edx
  _QWORD *v61; // rax
  __int128 v62; // [rsp-10h] [rbp-2C0h]
  __int128 v63; // [rsp-10h] [rbp-2C0h]
  __int128 v64; // [rsp+0h] [rbp-2B0h]
  __int64 v65; // [rsp+0h] [rbp-2B0h]
  __int64 v66; // [rsp+0h] [rbp-2B0h]
  __int64 v67; // [rsp+20h] [rbp-290h]
  __int64 v68; // [rsp+28h] [rbp-288h]
  __int64 v69; // [rsp+30h] [rbp-280h]
  int v70; // [rsp+30h] [rbp-280h]
  int v71; // [rsp+38h] [rbp-278h]
  __int64 v73; // [rsp+40h] [rbp-270h]
  __int64 v74; // [rsp+40h] [rbp-270h]
  int v75; // [rsp+40h] [rbp-270h]
  __int64 v76; // [rsp+48h] [rbp-268h]
  __int64 v77; // [rsp+98h] [rbp-218h] BYREF
  __int64 v78; // [rsp+A0h] [rbp-210h] BYREF
  int v79; // [rsp+A8h] [rbp-208h]
  unsigned __int16 v80; // [rsp+B0h] [rbp-200h] BYREF
  __int64 v81; // [rsp+B8h] [rbp-1F8h]
  __int64 v82; // [rsp+C0h] [rbp-1F0h] BYREF
  __int64 v83; // [rsp+C8h] [rbp-1E8h]
  _OWORD *v84; // [rsp+D0h] [rbp-1E0h] BYREF
  __int64 v85; // [rsp+D8h] [rbp-1D8h]
  _OWORD v86[8]; // [rsp+E0h] [rbp-1D0h] BYREF
  _BYTE *v87; // [rsp+160h] [rbp-150h] BYREF
  __int64 v88; // [rsp+168h] [rbp-148h]
  _BYTE v89[128]; // [rsp+170h] [rbp-140h] BYREF
  _BYTE *v90; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v91; // [rsp+1F8h] [rbp-B8h]
  _BYTE v92[176]; // [rsp+200h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 848);
  v78 = 0;
  v79 = v6;
  if ( v5 )
  {
    if ( &v78 != (__int64 *)(v5 + 48) )
    {
      v7 = *(_QWORD *)(v5 + 48);
      v78 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v78, v7, 1);
    }
  }
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v9 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v10 = *(__int64 **)(a2 + 8);
  v12 = *(_QWORD *)(v9 + 48) + 16LL * v11;
  v13 = *(_WORD *)v12;
  v81 = *(_QWORD *)(v12 + 8);
  v14 = *(_QWORD *)(a1 + 864);
  v80 = v13;
  v15 = sub_2E79000(*(__int64 **)(v14 + 40));
  LODWORD(v82) = sub_2D5BAE0(v8, v15, v10, 0);
  v84 = v86;
  v83 = v18;
  v69 = a3;
  v85 = 0x800000000LL;
  if ( a3 )
  {
    v19 = v86;
    if ( a3 > 8uLL )
    {
      sub_C8D5F0((__int64)&v84, v86, a3, 0x10u, v16, v17);
      v19 = &v84[(unsigned int)v85];
      for ( i = &v84[a3]; i != v19; ++v19 )
      {
LABEL_8:
        if ( v19 )
        {
          *(_QWORD *)v19 = 0;
          *((_DWORD *)v19 + 2) = 0;
        }
      }
    }
    else
    {
      i = &v86[a3];
      if ( v86 != i )
        goto LABEL_8;
    }
    v21 = 0;
    LODWORD(v85) = a3;
    do
    {
      v22 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (v21 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
      v24 = v23;
      v25 = v22;
      v26 = v21++;
      v27 = &v84[v26];
      *(_QWORD *)v27 = v25;
      *((_DWORD *)v27 + 2) = v24;
    }
    while ( a3 != v21 );
  }
  if ( (_WORD)v82 )
  {
    if ( (unsigned __int16)(v82 - 17) > 0x9Eu )
      goto LABEL_16;
  }
  else if ( !sub_30070D0((__int64)&v82) )
  {
    goto LABEL_16;
  }
  if ( a3 == 2 )
  {
    if ( v80 )
    {
      v52 = word_4456340[v80 - 1];
    }
    else
    {
      v77 = sub_3007240((__int64)&v80);
      v52 = v77;
    }
    *((_QWORD *)&v63 + 1) = (unsigned int)v85;
    *(_QWORD *)&v63 = v84;
    v53 = sub_33FC220(*(_QWORD *)(a1 + 864), 159, (unsigned int)&v78, v82, v83, v17, v63);
    v54 = *(_QWORD *)(a1 + 864);
    v71 = v55;
    v70 = v53;
    sub_9B9520((__int64)&v90, v52, 2);
    v56 = *(_QWORD *)(a1 + 864);
    v87 = 0;
    v74 = (__int64)v90;
    LODWORD(v88) = 0;
    v76 = (unsigned int)v91;
    v57 = sub_33F17F0(v56, 51, &v87, v82, v83);
    if ( v87 )
    {
      v67 = v57;
      v68 = v58;
      sub_B91220((__int64)&v87, (__int64)v87);
      v57 = v67;
      v58 = v68;
    }
    v59 = sub_33FCE10(v54, v82, v83, (unsigned int)&v78, v70, v71, v57, v58, v74, v76);
    v75 = v60;
    v87 = (_BYTE *)a2;
    v61 = sub_337DC20(a1 + 8, (__int64 *)&v87);
    *v61 = v59;
    *((_DWORD *)v61 + 2) = v75;
    v48 = (unsigned __int64)v90;
    if ( v90 != v92 )
      goto LABEL_34;
    goto LABEL_35;
  }
LABEL_16:
  v28 = v80;
  v29 = v81;
  v87 = v89;
  v88 = 0x800000000LL;
  if ( a3 > 8uLL )
  {
    v66 = v81;
    sub_C8D5F0((__int64)&v87, v89, a3, 0x10u, v81, v17);
    v50 = (__int64)v87;
    v51 = &v87[16 * a3];
    do
    {
      if ( v50 )
      {
        *(_WORD *)v50 = v28;
        *(_QWORD *)(v50 + 8) = v66;
      }
      v50 += 16;
    }
    while ( (_BYTE *)v50 != v51 );
    v31 = v87;
    LODWORD(v88) = a3;
  }
  else
  {
    v30 = (unsigned __int16 *)v89;
    v31 = v89;
    if ( a3 )
    {
      v49 = a3;
      do
      {
        *v30 = v28;
        v30 += 8;
        *((_QWORD *)v30 - 1) = v29;
        --v49;
      }
      while ( v49 );
      v31 = v87;
    }
    LODWORD(v88) = a3;
  }
  v32 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v64 = v84;
  *((_QWORD *)&v64 + 1) = (unsigned int)v85;
  v33 = sub_33E5830(v32, v31);
  v36 = sub_3411630(v32, 163, (unsigned int)&v78, v33, v34, v35, v64);
  v90 = v92;
  v38 = v92;
  v91 = 0x800000000LL;
  if ( a3 )
  {
    v39 = v92;
    v40 = v92;
    if ( a3 > 8uLL )
    {
      v65 = v36;
      sub_C8D5F0((__int64)&v90, v92, a3, 0x10u, v36, v37);
      v40 = v90;
      v36 = v65;
      v39 = &v90[16 * (unsigned int)v91];
    }
    v41 = &v40[16 * a3];
    if ( v39 != v41 )
    {
      do
      {
        if ( v39 )
        {
          *(_QWORD *)v39 = 0;
          *((_DWORD *)v39 + 2) = 0;
        }
        v39 += 16;
      }
      while ( v41 != v39 );
      v40 = v90;
    }
    LODWORD(v91) = a3;
    v42 = 0;
    while ( 1 )
    {
      v43 = &v40[16 * v42];
      *((_DWORD *)v43 + 2) = v42++;
      *(_QWORD *)v43 = v36;
      if ( v42 == a3 )
        break;
      v40 = v90;
    }
    v38 = v90;
    v69 = (unsigned int)v91;
  }
  *((_QWORD *)&v62 + 1) = v69;
  *(_QWORD *)&v62 = v38;
  v44 = sub_33FC220(*(_QWORD *)(a1 + 864), 159, (unsigned int)&v78, v82, v83, v37, v62);
  v77 = a2;
  v73 = v44;
  v46 = v45;
  v47 = sub_337DC20(a1 + 8, &v77);
  *v47 = v73;
  *((_DWORD *)v47 + 2) = v46;
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
  v48 = (unsigned __int64)v87;
  if ( v87 != v89 )
LABEL_34:
    _libc_free(v48);
LABEL_35:
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
  if ( v78 )
    sub_B91220((__int64)&v78, v78);
}
