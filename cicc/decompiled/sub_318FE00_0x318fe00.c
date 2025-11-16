// Function: sub_318FE00
// Address: 0x318fe00
//
__int64 __fastcall sub_318FE00(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  unsigned int *v10; // rax
  int v11; // ecx
  unsigned int *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rax
  int v18; // ecx
  unsigned int *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rsi
  __int64 v23; // rcx
  int v24; // eax
  int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rcx
  int v32; // eax
  int v33; // eax
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // r14
  __int64 v40; // r12
  int v41; // eax
  int v42; // eax
  unsigned int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // r12
  __int64 v48; // rbx
  int v49; // eax
  int v50; // eax
  unsigned int v51; // edx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  unsigned __int64 v56; // rsi
  unsigned __int64 v57; // rsi
  __int64 v58; // [rsp+0h] [rbp-130h]
  __int64 v59; // [rsp+0h] [rbp-130h]
  _QWORD v61[4]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v62; // [rsp+60h] [rbp-D0h]
  unsigned int *v63; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v64; // [rsp+78h] [rbp-B8h]
  _BYTE v65[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+A0h] [rbp-90h]
  __int64 v67; // [rsp+A8h] [rbp-88h]
  __int16 v68; // [rsp+B0h] [rbp-80h]
  __int64 v69; // [rsp+B8h] [rbp-78h]
  void **v70; // [rsp+C0h] [rbp-70h]
  void **v71; // [rsp+C8h] [rbp-68h]
  __int64 v72; // [rsp+D0h] [rbp-60h]
  int v73; // [rsp+D8h] [rbp-58h]
  __int16 v74; // [rsp+DCh] [rbp-54h]
  char v75; // [rsp+DEh] [rbp-52h]
  __int64 v76; // [rsp+E0h] [rbp-50h]
  __int64 v77; // [rsp+E8h] [rbp-48h]
  void *v78; // [rsp+F0h] [rbp-40h] BYREF
  void *v79; // [rsp+F8h] [rbp-38h] BYREF

  v5 = *(_QWORD *)(a4 + 56);
  v66 = a4;
  v69 = sub_AA48A0(a4);
  v70 = &v78;
  v71 = &v79;
  v63 = (unsigned int *)v65;
  v78 = &unk_49DA100;
  v64 = 0x200000000LL;
  v72 = 0;
  v79 = &unk_49DA0B0;
  v73 = 0;
  v74 = 512;
  v75 = 7;
  v76 = 0;
  v77 = 0;
  v67 = v5;
  v68 = 1;
  if ( v5 != v66 + 48 )
  {
    if ( v5 )
      v5 -= 24;
    v6 = *(_QWORD *)sub_B46C60(v5);
    v61[0] = v6;
    if ( v6 && (sub_B96E90((__int64)v61, v6, 1), (v9 = v61[0]) != 0) )
    {
      v10 = v63;
      v11 = v64;
      v12 = &v63[4 * (unsigned int)v64];
      if ( v63 != v12 )
      {
        while ( 1 )
        {
          v7 = *v10;
          if ( !(_DWORD)v7 )
            break;
          v10 += 4;
          if ( v12 == v10 )
            goto LABEL_68;
        }
        *((_QWORD *)v10 + 1) = v61[0];
LABEL_11:
        sub_B91220((__int64)v61, v9);
        goto LABEL_14;
      }
LABEL_68:
      if ( (unsigned int)v64 >= (unsigned __int64)HIDWORD(v64) )
      {
        v57 = (unsigned int)v64 + 1LL;
        if ( HIDWORD(v64) < v57 )
        {
          sub_C8D5F0((__int64)&v63, v65, v57, 0x10u, v7, v8);
          v12 = &v63[4 * (unsigned int)v64];
        }
        *(_QWORD *)v12 = 0;
        *((_QWORD *)v12 + 1) = v9;
        v9 = v61[0];
        LODWORD(v64) = v64 + 1;
      }
      else
      {
        if ( v12 )
        {
          *v12 = 0;
          *((_QWORD *)v12 + 1) = v9;
          v11 = v64;
          v9 = v61[0];
        }
        LODWORD(v64) = v11 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v63, 0);
      v9 = v61[0];
    }
    if ( !v9 )
      goto LABEL_14;
    goto LABEL_11;
  }
LABEL_14:
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
  v61[0] = v13;
  if ( v13 && (sub_B96E90((__int64)v61, v13, 1), (v16 = v61[0]) != 0) )
  {
    v17 = (__int64)v63;
    v18 = v64;
    v19 = &v63[4 * (unsigned int)v64];
    if ( v63 != v19 )
    {
      while ( *(_DWORD *)v17 )
      {
        v17 += 16;
        if ( v19 == (unsigned int *)v17 )
          goto LABEL_64;
      }
      *(_QWORD *)(v17 + 8) = v61[0];
      goto LABEL_21;
    }
LABEL_64:
    if ( (unsigned int)v64 >= (unsigned __int64)HIDWORD(v64) )
    {
      v56 = (unsigned int)v64 + 1LL;
      if ( HIDWORD(v64) < v56 )
      {
        sub_C8D5F0((__int64)&v63, v65, v56, 0x10u, v14, v15);
        v19 = &v63[4 * (unsigned int)v64];
      }
      *(_QWORD *)v19 = 0;
      *((_QWORD *)v19 + 1) = v16;
      v16 = v61[0];
      LODWORD(v64) = v64 + 1;
    }
    else
    {
      if ( v19 )
      {
        *v19 = 0;
        *((_QWORD *)v19 + 1) = v16;
        v18 = v64;
        v16 = v61[0];
      }
      LODWORD(v64) = v18 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v63, 0);
    v16 = v61[0];
  }
  if ( v16 )
LABEL_21:
    sub_B91220((__int64)v61, v16);
  v20 = *(_QWORD *)(a1 + 8);
  v62 = 257;
  v21 = sub_D5C860((__int64 *)&v63, *(_QWORD *)(v20 + 8), 2, (__int64)v61);
  v22 = *a2;
  v23 = a2[1];
  v24 = *(_DWORD *)(v21 + 4) & 0x7FFFFFF;
  if ( v24 == *(_DWORD *)(v21 + 72) )
  {
    v59 = a2[1];
    sub_B48D90(v21);
    v23 = v59;
    v24 = *(_DWORD *)(v21 + 4) & 0x7FFFFFF;
  }
  v25 = (v24 + 1) & 0x7FFFFFF;
  v26 = v25 | *(_DWORD *)(v21 + 4) & 0xF8000000;
  v27 = *(_QWORD *)(v21 - 8) + 32LL * (unsigned int)(v25 - 1);
  *(_DWORD *)(v21 + 4) = v26;
  if ( *(_QWORD *)v27 )
  {
    v28 = *(_QWORD *)(v27 + 8);
    **(_QWORD **)(v27 + 16) = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(v27 + 16);
  }
  *(_QWORD *)v27 = v23;
  if ( v23 )
  {
    v29 = *(_QWORD *)(v23 + 16);
    *(_QWORD *)(v27 + 8) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = v27 + 8;
    *(_QWORD *)(v27 + 16) = v23 + 16;
    *(_QWORD *)(v23 + 16) = v27;
  }
  *(_QWORD *)(*(_QWORD *)(v21 - 8) + 32LL * *(unsigned int *)(v21 + 72)
                                   + 8LL * ((*(_DWORD *)(v21 + 4) & 0x7FFFFFFu) - 1)) = v22;
  v30 = *a3;
  v31 = a3[1];
  v32 = *(_DWORD *)(v21 + 4) & 0x7FFFFFF;
  if ( v32 == *(_DWORD *)(v21 + 72) )
  {
    v58 = a3[1];
    sub_B48D90(v21);
    v31 = v58;
    v32 = *(_DWORD *)(v21 + 4) & 0x7FFFFFF;
  }
  v33 = (v32 + 1) & 0x7FFFFFF;
  v34 = v33 | *(_DWORD *)(v21 + 4) & 0xF8000000;
  v35 = *(_QWORD *)(v21 - 8) + 32LL * (unsigned int)(v33 - 1);
  *(_DWORD *)(v21 + 4) = v34;
  if ( *(_QWORD *)v35 )
  {
    v36 = *(_QWORD *)(v35 + 8);
    **(_QWORD **)(v35 + 16) = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 16) = *(_QWORD *)(v35 + 16);
  }
  *(_QWORD *)v35 = v31;
  if ( v31 )
  {
    v37 = *(_QWORD *)(v31 + 16);
    *(_QWORD *)(v35 + 8) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = v35 + 8;
    *(_QWORD *)(v35 + 16) = v31 + 16;
    *(_QWORD *)(v31 + 16) = v35;
  }
  *(_QWORD *)(*(_QWORD *)(v21 - 8) + 32LL * *(unsigned int *)(v21 + 72)
                                   + 8LL * ((*(_DWORD *)(v21 + 4) & 0x7FFFFFFu) - 1)) = v30;
  v62 = 257;
  v38 = sub_D5C860((__int64 *)&v63, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL), 2, (__int64)v61);
  v39 = *a2;
  v40 = a2[2];
  v41 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  if ( v41 == *(_DWORD *)(v38 + 72) )
  {
    sub_B48D90(v38);
    v41 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  }
  v42 = (v41 + 1) & 0x7FFFFFF;
  v43 = v42 | *(_DWORD *)(v38 + 4) & 0xF8000000;
  v44 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v42 - 1);
  *(_DWORD *)(v38 + 4) = v43;
  if ( *(_QWORD *)v44 )
  {
    v45 = *(_QWORD *)(v44 + 8);
    **(_QWORD **)(v44 + 16) = v45;
    if ( v45 )
      *(_QWORD *)(v45 + 16) = *(_QWORD *)(v44 + 16);
  }
  *(_QWORD *)v44 = v40;
  if ( v40 )
  {
    v46 = *(_QWORD *)(v40 + 16);
    *(_QWORD *)(v44 + 8) = v46;
    if ( v46 )
      *(_QWORD *)(v46 + 16) = v44 + 8;
    *(_QWORD *)(v44 + 16) = v40 + 16;
    *(_QWORD *)(v40 + 16) = v44;
  }
  *(_QWORD *)(*(_QWORD *)(v38 - 8) + 32LL * *(unsigned int *)(v38 + 72)
                                   + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v39;
  v47 = *a3;
  v48 = a3[2];
  v49 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  if ( v49 == *(_DWORD *)(v38 + 72) )
  {
    sub_B48D90(v38);
    v49 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  }
  v50 = (v49 + 1) & 0x7FFFFFF;
  v51 = v50 | *(_DWORD *)(v38 + 4) & 0xF8000000;
  v52 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v50 - 1);
  *(_DWORD *)(v38 + 4) = v51;
  if ( *(_QWORD *)v52 )
  {
    v53 = *(_QWORD *)(v52 + 8);
    **(_QWORD **)(v52 + 16) = v53;
    if ( v53 )
      *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
  }
  *(_QWORD *)v52 = v48;
  if ( v48 )
  {
    v54 = *(_QWORD *)(v48 + 16);
    *(_QWORD *)(v52 + 8) = v54;
    if ( v54 )
      *(_QWORD *)(v54 + 16) = v52 + 8;
    *(_QWORD *)(v52 + 16) = v48 + 16;
    *(_QWORD *)(v48 + 16) = v52;
  }
  *(_QWORD *)(*(_QWORD *)(v38 - 8) + 32LL * *(unsigned int *)(v38 + 72)
                                   + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v47;
  nullsub_61();
  v78 = &unk_49DA100;
  nullsub_63();
  if ( v63 != (unsigned int *)v65 )
    _libc_free((unsigned __int64)v63);
  return v21;
}
