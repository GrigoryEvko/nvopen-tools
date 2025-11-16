// Function: sub_33DC4D0
// Address: 0x33dc4d0
//
__int64 __fastcall sub_33DC4D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // r12
  unsigned __int16 *v9; // rcx
  int v10; // eax
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // r13
  unsigned int v17; // r14d
  unsigned __int64 v18; // rax
  int v19; // eax
  bool v20; // cc
  unsigned __int64 v21; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // r14
  __int64 v28; // r12
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // rsi
  int v32; // eax
  unsigned int v33; // eax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  int v36; // eax
  int v37; // r14d
  _QWORD *v38; // r15
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // r13
  unsigned int v44; // eax
  int v45; // eax
  __int64 v46; // r10
  unsigned __int64 *v47; // r13
  unsigned __int64 v48; // rax
  __int64 v49; // [rsp+8h] [rbp-F8h]
  int v50; // [rsp+14h] [rbp-ECh]
  unsigned __int64 v51; // [rsp+18h] [rbp-E8h]
  __int64 v52; // [rsp+20h] [rbp-E0h]
  int v53; // [rsp+28h] [rbp-D8h]
  __int64 v54; // [rsp+38h] [rbp-C8h]
  __int64 v55; // [rsp+40h] [rbp-C0h]
  unsigned int v56; // [rsp+48h] [rbp-B8h]
  unsigned int v57; // [rsp+4Ch] [rbp-B4h]
  __int64 v60; // [rsp+58h] [rbp-A8h]
  __int64 v61; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v62; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v63; // [rsp+68h] [rbp-98h]
  unsigned __int64 v64; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v65; // [rsp+78h] [rbp-88h]
  __int64 v66; // [rsp+80h] [rbp-80h]
  __int64 v67; // [rsp+88h] [rbp-78h]
  unsigned __int64 v68; // [rsp+90h] [rbp-70h] BYREF
  __int64 v69; // [rsp+98h] [rbp-68h]
  unsigned __int64 v70; // [rsp+A0h] [rbp-60h]
  unsigned int v71; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v72; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-48h]
  __int64 v74; // [rsp+C0h] [rbp-40h]
  int v75; // [rsp+C8h] [rbp-38h]

  v8 = a1;
  v9 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * a4);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v68) = v10;
  v69 = v11;
  if ( !(_WORD)v10 )
  {
    if ( !sub_30070B0((__int64)&v68) )
    {
      v73 = v11;
      LOWORD(v72) = 0;
      goto LABEL_16;
    }
    LOWORD(v10) = sub_3009970((__int64)&v68, a2, v39, v40, v41);
LABEL_15:
    LOWORD(v72) = v10;
    v73 = v23;
    if ( (_WORD)v10 )
      goto LABEL_4;
LABEL_16:
    v24 = sub_3007260((__int64)&v72);
    v12 = *(_QWORD *)(a3 + 40);
    v26 = v25;
    v66 = v24;
    LODWORD(v13) = v24;
    v14 = *(_QWORD *)(v12 + 40);
    v67 = v26;
    v15 = *(_DWORD *)(v14 + 24);
    if ( v15 == 35 )
      goto LABEL_7;
LABEL_17:
    if ( v15 == 11 )
      goto LABEL_7;
    if ( v15 == 156 && *(_DWORD *)(v14 + 64) )
    {
      v55 = *(unsigned int *)(v14 + 64);
      v51 = (unsigned int)v13;
      v54 = v14;
      v53 = v13;
      v13 = 0;
      v57 = *(_DWORD *)(a5 + 8);
      v52 = v12;
      v27 = 0;
      v28 = *(_QWORD *)a5;
      v56 = a6;
      v29 = 0;
      do
      {
        v30 = v28;
        if ( v57 > 0x40 )
          v30 = *(_QWORD *)(v28 + 8LL * ((unsigned int)v29 >> 6));
        if ( (v30 & (1LL << v29)) != 0 )
        {
          v31 = *(_QWORD *)(*(_QWORD *)(v54 + 40) + 40 * v29);
          v32 = *(_DWORD *)(v31 + 24);
          if ( v32 != 11 && v32 != 35 )
          {
            LODWORD(v13) = v53;
            v12 = v52;
            v8 = a1;
            a6 = v56;
            goto LABEL_27;
          }
          v46 = *(_QWORD *)(v31 + 96);
          v47 = (unsigned __int64 *)(v46 + 24);
          if ( *(_DWORD *)(v46 + 32) <= 0x40u )
          {
            v48 = *(_QWORD *)(v46 + 24);
          }
          else
          {
            v49 = *(_QWORD *)(v31 + 96);
            v50 = *(_DWORD *)(v46 + 32);
            if ( v50 - (unsigned int)sub_C444A0(v46 + 24) > 0x40 )
              goto LABEL_77;
            v48 = **(_QWORD **)(v49 + 24);
          }
          if ( v48 >= v51 )
          {
LABEL_77:
            v8 = a1;
            *(_BYTE *)(a1 + 32) = 0;
            return v8;
          }
          if ( v13 )
          {
            if ( (int)sub_C49970(v13, v47) > 0 )
              v13 = (__int64)v47;
          }
          else
          {
            v13 = (__int64)v47;
          }
          if ( v27 )
          {
            if ( (int)sub_C49970(v27, v47) < 0 )
              v27 = (__int64)v47;
          }
          else
          {
            v27 = (__int64)v47;
          }
        }
        ++v29;
      }
      while ( v55 != v29 );
      v42 = v13;
      v43 = v27;
      LODWORD(v13) = v53;
      v12 = v52;
      v8 = a1;
      a6 = v56;
      if ( v42 )
      {
        if ( v43 )
        {
          v63 = *(_DWORD *)(v43 + 8);
          if ( v63 > 0x40 )
          {
            v61 = v42;
            sub_C43780((__int64)&v62, (const void **)v43);
            v42 = v61;
          }
          else
          {
            v62 = *(_QWORD *)v43;
          }
          v60 = v42;
          sub_C46A40((__int64)&v62, 1);
          v44 = v63;
          v63 = 0;
          v65 = v44;
          v64 = v62;
          LODWORD(v69) = *(_DWORD *)(v60 + 8);
          if ( (unsigned int)v69 > 0x40 )
            sub_C43780((__int64)&v68, (const void **)v60);
          else
            v68 = *(_QWORD *)v60;
          sub_AADC30((__int64)&v72, (__int64)&v68, (__int64 *)&v64);
          v45 = v73;
          v20 = (unsigned int)v69 <= 0x40;
          *(_BYTE *)(a1 + 32) = 1;
          *(_DWORD *)(a1 + 8) = v45;
          *(_QWORD *)a1 = v72;
          *(_DWORD *)(a1 + 24) = v75;
          *(_QWORD *)(a1 + 16) = v74;
          if ( !v20 && v68 )
            j_j___libc_free_0_0(v68);
          if ( v65 > 0x40 && v64 )
            j_j___libc_free_0_0(v64);
          if ( v63 > 0x40 )
          {
            v21 = v62;
            if ( v62 )
              goto LABEL_12;
          }
          return v8;
        }
      }
    }
LABEL_27:
    sub_33D4EF0((__int64)&v68, a2, *(_QWORD *)(v12 + 40), *(_QWORD *)(v12 + 48), a5, a6);
    v33 = v69;
    LODWORD(v73) = v69;
    if ( (unsigned int)v69 > 0x40 )
    {
      sub_C43780((__int64)&v72, (const void **)&v68);
      v33 = v73;
      if ( (unsigned int)v73 > 0x40 )
      {
        sub_C43D10((__int64)&v72);
        v37 = v73;
        v38 = (_QWORD *)v72;
        v65 = v73;
        v64 = v72;
        if ( (unsigned int)v73 <= 0x40 )
        {
          if ( v72 < (unsigned int)v13 )
            goto LABEL_32;
        }
        else
        {
          if ( v37 - (unsigned int)sub_C444A0((__int64)&v64) <= 0x40 && *v38 < (unsigned __int64)(unsigned int)v13 )
          {
            if ( v38 )
              j_j___libc_free_0_0((unsigned __int64)v38);
LABEL_32:
            sub_AAF050((__int64)&v72, (__int64)&v68, 0);
            v36 = v73;
            v20 = v71 <= 0x40;
            *(_BYTE *)(v8 + 32) = 1;
            *(_DWORD *)(v8 + 8) = v36;
            *(_QWORD *)v8 = v72;
            *(_DWORD *)(v8 + 24) = v75;
            *(_QWORD *)(v8 + 16) = v74;
            if ( v20 )
              goto LABEL_35;
            goto LABEL_33;
          }
          if ( v38 )
            j_j___libc_free_0_0((unsigned __int64)v38);
        }
LABEL_43:
        v20 = v71 <= 0x40;
        *(_BYTE *)(v8 + 32) = 0;
        if ( v20 )
        {
          if ( (unsigned int)v69 <= 0x40 )
            return v8;
LABEL_36:
          v21 = v68;
          if ( !v68 )
            return v8;
          goto LABEL_12;
        }
LABEL_33:
        if ( v70 )
          j_j___libc_free_0_0(v70);
LABEL_35:
        if ( (unsigned int)v69 <= 0x40 )
          return v8;
        goto LABEL_36;
      }
      v34 = v72;
    }
    else
    {
      v34 = v68;
    }
    v65 = v33;
    v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v33) & ~v34;
    if ( !v33 )
      v35 = 0;
    v64 = v35;
    if ( (unsigned int)v13 > v35 )
      goto LABEL_32;
    goto LABEL_43;
  }
  if ( (unsigned __int16)(v10 - 17) <= 0xD3u )
  {
    LOWORD(v10) = word_4456580[v10 - 1];
    v23 = 0;
    goto LABEL_15;
  }
  LOWORD(v72) = v10;
  v73 = v11;
LABEL_4:
  if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
    BUG();
  v12 = *(_QWORD *)(a3 + 40);
  v13 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
  v14 = *(_QWORD *)(v12 + 40);
  v15 = *(_DWORD *)(v14 + 24);
  if ( v15 != 35 )
    goto LABEL_17;
LABEL_7:
  v16 = *(_QWORD *)(v14 + 96);
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 > 0x40 )
  {
    if ( v17 - (unsigned int)sub_C444A0(v16 + 24) <= 0x40
      && (unsigned __int64)(unsigned int)v13 > **(_QWORD **)(v16 + 24) )
    {
      LODWORD(v69) = v17;
      sub_C43780((__int64)&v68, (const void **)(v16 + 24));
      goto LABEL_10;
    }
LABEL_48:
    *(_BYTE *)(a1 + 32) = 0;
    return v8;
  }
  v18 = *(_QWORD *)(v16 + 24);
  if ( (unsigned int)v13 <= v18 )
    goto LABEL_48;
  LODWORD(v69) = *(_DWORD *)(v16 + 32);
  v68 = v18;
LABEL_10:
  sub_AADBC0((__int64)&v72, (__int64 *)&v68);
  v19 = v73;
  v20 = (unsigned int)v69 <= 0x40;
  *(_BYTE *)(a1 + 32) = 1;
  *(_DWORD *)(a1 + 8) = v19;
  *(_QWORD *)a1 = v72;
  *(_DWORD *)(a1 + 24) = v75;
  *(_QWORD *)(a1 + 16) = v74;
  if ( !v20 )
  {
    v21 = v68;
    if ( v68 )
LABEL_12:
      j_j___libc_free_0_0(v21);
  }
  return v8;
}
