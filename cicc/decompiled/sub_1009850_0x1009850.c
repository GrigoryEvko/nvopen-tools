// Function: sub_1009850
// Address: 0x1009850
//
unsigned __int8 *__fastcall sub_1009850(__int64 a1, __int64 a2, char a3, __m128i *a4, char a5, char a6)
{
  __int64 v6; // r15
  __int64 v8; // r12
  unsigned __int8 *v10; // r9
  bool v12; // r14
  unsigned int v13; // eax
  unsigned __int8 *v14; // r9
  unsigned int v15; // eax
  void *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int16 v22; // r14
  __int16 v23; // r14
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // r13
  void *v27; // rax
  __int64 v28; // rax
  bool v29; // al
  __int64 v30; // rax
  void **v31; // rax
  __int64 v32; // r8
  bool v33; // dl
  void *v34; // rax
  _BYTE *v35; // rcx
  __int64 v36; // rax
  _BYTE *v37; // rax
  __int64 v38; // r8
  bool v39; // dl
  void *v40; // rax
  _BYTE *v41; // rcx
  __int64 v42; // rsi
  void **v43; // rax
  void **v44; // rcx
  char v45; // al
  void *v46; // rax
  _BYTE *v47; // rcx
  __int64 v48; // rsi
  void **v49; // rax
  void **v50; // rcx
  char v51; // al
  void *v52; // rax
  _BYTE *v53; // rcx
  int v54; // [rsp+Ch] [rbp-64h]
  int v55; // [rsp+Ch] [rbp-64h]
  unsigned __int8 *v56; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v57; // [rsp+10h] [rbp-60h]
  void **v58; // [rsp+10h] [rbp-60h]
  bool v59; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v60; // [rsp+10h] [rbp-60h]
  char v61; // [rsp+18h] [rbp-58h]
  bool v62; // [rsp+18h] [rbp-58h]
  bool v63; // [rsp+18h] [rbp-58h]
  bool v64; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v65; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v68; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v69; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v70; // [rsp+28h] [rbp-48h]
  __int64 v71; // [rsp+28h] [rbp-48h]
  void **v72; // [rsp+28h] [rbp-48h]
  __int64 v73; // [rsp+28h] [rbp-48h]
  _BYTE *v74; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v75; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v76; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v77; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v78; // [rsp+28h] [rbp-48h]
  unsigned __int8 *v79; // [rsp+28h] [rbp-48h]
  void **v80; // [rsp+28h] [rbp-48h]
  unsigned __int64 v81; // [rsp+30h] [rbp-40h] BYREF
  __int64 v82; // [rsp+38h] [rbp-38h]

  v6 = a2;
  v8 = a1;
  v81 = a1;
  v82 = a2;
  v10 = sub_1003820((__int64 *)&v81, 2, a3, (__int64)a4, a5, a6);
  if ( v10 )
    return v10;
  v12 = a6 == 1 && a5 == 0;
  if ( !v12 )
    return v10;
  v81 = 0x3FF0000000000000LL;
  v13 = sub_1009690((double *)&v81, a1);
  v14 = 0;
  if ( !(_BYTE)v13 )
  {
    if ( *(_BYTE *)a1 == 18 )
    {
      v27 = sub_C33340();
      v14 = 0;
      if ( *(void **)(a1 + 24) == v27 )
        v28 = *(_QWORD *)(a1 + 32);
      else
        v28 = a1 + 24;
      v29 = (*(_BYTE *)(v28 + 20) & 7) == 3;
      goto LABEL_32;
    }
    v63 = v13;
    v73 = *(_QWORD *)(a1 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v73 + 8) - 17 <= 1 && *(_BYTE *)a1 <= 0x15u )
    {
      v37 = sub_AD7630(a1, 0, v13);
      v14 = 0;
      v38 = v73;
      v39 = v63;
      if ( v37 )
      {
        v74 = v37;
        if ( *v37 == 18 )
        {
          v40 = sub_C33340();
          v14 = 0;
          if ( *((void **)v74 + 3) == v40 )
            v41 = (_BYTE *)*((_QWORD *)v74 + 4);
          else
            v41 = v74 + 24;
          v29 = (v41[20] & 7) == 3;
LABEL_32:
          if ( v29 )
            goto LABEL_5;
          goto LABEL_33;
        }
      }
      if ( *(_BYTE *)(v38 + 8) == 17 )
      {
        v55 = *(_DWORD *)(v38 + 32);
        if ( v55 )
        {
          v48 = 0;
          while ( 1 )
          {
            v79 = v14;
            v59 = v39;
            v49 = (void **)sub_AD69F0((unsigned __int8 *)a1, v48);
            v14 = v79;
            v50 = v49;
            if ( !v49 )
              break;
            v51 = *(_BYTE *)v49;
            v80 = v50;
            v39 = v59;
            if ( v51 != 13 )
            {
              if ( v51 != 18 )
                break;
              v60 = v14;
              v52 = sub_C33340();
              v14 = v60;
              v53 = v80[3] == v52 ? v80[4] : v80 + 3;
              if ( (v53[20] & 7) != 3 )
                break;
              v39 = v12;
            }
            v48 = (unsigned int)(v48 + 1);
            if ( v55 == (_DWORD)v48 )
            {
              if ( v39 )
                goto LABEL_5;
              break;
            }
          }
        }
      }
    }
LABEL_33:
    v30 = v6;
    v6 = a1;
    v8 = v30;
  }
LABEL_5:
  v68 = v14;
  v81 = 0x3FF0000000000000LL;
  v15 = sub_1009690((double *)&v81, v8);
  v10 = (unsigned __int8 *)v6;
  if ( (_BYTE)v15 )
    return v10;
  v10 = v68;
  if ( *(_BYTE *)v8 == 18 )
  {
    v16 = sub_C33340();
    v10 = v68;
    if ( *(void **)(v8 + 24) == v16 )
      v17 = *(_QWORD *)(v8 + 32);
    else
      v17 = v8 + 24;
    if ( (*(_BYTE *)(v17 + 20) & 7) != 3 )
      goto LABEL_41;
  }
  else
  {
    v62 = v15;
    v71 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v71 + 8) - 17 > 1 || *(_BYTE *)v8 > 0x15u )
      goto LABEL_41;
    v56 = v10;
    v31 = (void **)sub_AD7630(v8, 0, v15);
    v10 = v56;
    v32 = v71;
    v33 = v62;
    if ( !v31 || (v72 = v31, *(_BYTE *)v31 != 18) )
    {
      if ( *(_BYTE *)(v32 + 8) == 17 )
      {
        v54 = *(_DWORD *)(v32 + 32);
        if ( v54 )
        {
          v42 = 0;
          while ( 1 )
          {
            v57 = v10;
            v64 = v33;
            v43 = (void **)sub_AD69F0((unsigned __int8 *)v8, v42);
            v10 = v57;
            v44 = v43;
            if ( !v43 )
              break;
            v45 = *(_BYTE *)v43;
            v58 = v44;
            v33 = v64;
            if ( v45 != 13 )
            {
              if ( v45 != 18 )
                goto LABEL_41;
              v65 = v10;
              v46 = sub_C33340();
              v10 = v65;
              v47 = v58[3] == v46 ? v58[4] : v58 + 3;
              if ( (v47[20] & 7) != 3 )
                goto LABEL_41;
              v33 = v12;
            }
            v42 = (unsigned int)(v42 + 1);
            if ( v54 == (_DWORD)v42 )
            {
              if ( v33 )
                goto LABEL_10;
              goto LABEL_41;
            }
          }
        }
      }
      goto LABEL_41;
    }
    v34 = sub_C33340();
    v10 = v56;
    v35 = v72[3] == v34 ? v72[4] : v72 + 3;
    if ( (v35[20] & 7) != 3 )
    {
LABEL_41:
      if ( v6 == v8 && *(_BYTE *)v6 == 85 )
      {
        v36 = *(_QWORD *)(v6 - 32);
        if ( v36 )
        {
          if ( !*(_BYTE *)v36
            && *(_QWORD *)(v36 + 24) == *(_QWORD *)(v6 + 80)
            && *(_DWORD *)(v36 + 36) == 335
            && *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF))
            && (a3 & 1) != 0
            && (a3 & 0xA) == 0xA )
          {
            return *(unsigned __int8 **)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
          }
        }
      }
      return v10;
    }
  }
LABEL_10:
  v18 = *(_QWORD *)(v6 + 8);
  if ( (a3 & 0xA) != 0xA )
  {
    if ( *(_BYTE *)(v18 + 8) == 17 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      LODWORD(v82) = v19;
      if ( v19 > 0x40 )
      {
        v78 = v10;
        sub_C43690((__int64)&v81, -1, 1);
        v10 = v78;
      }
      else
      {
        v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
        if ( !v19 )
          v20 = 0;
        v81 = v20;
      }
    }
    else
    {
      LODWORD(v82) = 1;
      v81 = 1;
    }
    if ( (a3 & 2) != 0 )
    {
      if ( (a3 & 4) == 0 )
      {
        v77 = v10;
        v21 = sub_9B3E70((__int64 *)v6, (__int64 *)&v81, 516, 0, a4);
        v10 = v77;
        v23 = v21 & 0x3FC;
        goto LABEL_20;
      }
      v69 = v10;
      v21 = sub_9B3E70((__int64 *)v6, (__int64 *)&v81, 0, 0, a4);
      v10 = v69;
      v22 = v21 & 0x3FC;
    }
    else
    {
      if ( (a3 & 4) == 0 )
      {
        v76 = v10;
        v21 = sub_9B3E70((__int64 *)v6, (__int64 *)&v81, 519, 0, a4);
        v10 = v76;
        v23 = v21;
        goto LABEL_20;
      }
      v75 = v10;
      v21 = sub_9B3E70((__int64 *)v6, (__int64 *)&v81, 3, 0, a4);
      v10 = v75;
      v22 = v21;
    }
    v23 = v22 & 0x1FB;
LABEL_20:
    v24 = v21;
    v25 = v21 >> 40;
    v26 = HIDWORD(v24);
    if ( (unsigned int)v82 > 0x40 && v81 )
    {
      v61 = v25;
      v70 = v10;
      j_j___libc_free_0_0(v81);
      LOBYTE(v25) = v61;
      v10 = v70;
    }
    if ( (v23 & 0x207) == 0 && (_BYTE)v25 )
    {
      if ( !(_BYTE)v26 )
        return (unsigned __int8 *)v8;
      if ( *(_BYTE *)v8 <= 0x15u )
        return (unsigned __int8 *)sub_96E680(12, v8);
      return v10;
    }
    goto LABEL_41;
  }
  return sub_AD9290(v18, 0);
}
