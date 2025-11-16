// Function: sub_2576E80
// Address: 0x2576e80
//
__int64 __fastcall sub_2576E80(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, unsigned __int8 a6)
{
  int v6; // r10d
  unsigned int v11; // r15d
  int v13; // r15d
  __int64 v14; // r11
  int v15; // ecx
  __int64 *v16; // r9
  __int64 *v17; // rsi
  int v18; // edx
  int v19; // r15d
  unsigned int i; // ecx
  __int64 ***v21; // r8
  __int64 **v22; // rdi
  char v23; // al
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 **v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r9
  __int64 v30; // r8
  unsigned __int8 v31; // r10
  __int64 v32; // rdi
  __int64 v33; // r8
  int v34; // eax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 *v37; // r13
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // r9
  __int64 v45; // r8
  unsigned __int8 v46; // r10
  __int64 v47; // rdx
  __int64 *v48; // rcx
  __int64 v49; // rcx
  unsigned __int64 v50; // rsi
  char v51; // al
  __int64 *v52; // rax
  char v53; // al
  _QWORD *v54; // r8
  int v55; // eax
  unsigned __int8 v56; // [rsp+Ch] [rbp-94h]
  unsigned __int8 v57; // [rsp+Ch] [rbp-94h]
  unsigned __int8 v58; // [rsp+10h] [rbp-90h]
  unsigned __int8 v59; // [rsp+10h] [rbp-90h]
  _QWORD **v60; // [rsp+10h] [rbp-90h]
  __int64 v61; // [rsp+10h] [rbp-90h]
  int v62; // [rsp+18h] [rbp-88h]
  int v63; // [rsp+18h] [rbp-88h]
  unsigned int v64; // [rsp+20h] [rbp-80h]
  unsigned int v65; // [rsp+20h] [rbp-80h]
  int v66; // [rsp+28h] [rbp-78h]
  __int64 v67; // [rsp+28h] [rbp-78h]
  int v69; // [rsp+28h] [rbp-78h]
  char v71; // [rsp+38h] [rbp-68h]
  __int64 v72; // [rsp+38h] [rbp-68h]
  __int64 v73; // [rsp+38h] [rbp-68h]
  __int64 v74; // [rsp+38h] [rbp-68h]
  __int64 *v75; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v76; // [rsp+48h] [rbp-58h] BYREF
  __int64 *v77[10]; // [rsp+50h] [rbp-50h] BYREF

  v6 = a5;
  *(_DWORD *)(a4 + 24) = a3;
  if ( !a6 )
    goto LABEL_2;
  v13 = *(_DWORD *)(a1 + 192);
  if ( !v13 )
    goto LABEL_2;
  v14 = *(_QWORD *)(a1 + 176);
  if ( !byte_4FEF240[0] )
  {
    v73 = *(_QWORD *)(a1 + 176);
    v34 = sub_2207590((__int64)byte_4FEF240);
    v14 = v73;
    v6 = a5;
    if ( v34 )
    {
      qword_4FEF260 = -4096;
      unk_4FEF268 = -4096;
      qword_4FEF270 = 0;
      unk_4FEF278 = 0;
      sub_2207640((__int64)byte_4FEF240);
      v6 = a5;
      v14 = v73;
    }
  }
  v15 = *(_DWORD *)(a4 + 28);
  v16 = *(__int64 **)a4;
  v17 = *(__int64 **)(a4 + 8);
  if ( !v15 )
  {
    v32 = *(_QWORD *)(a4 + 16);
    v33 = 0;
    if ( v32 )
    {
      v17 = *(__int64 **)(a4 + 8);
      v33 = (unsigned int)sub_253B7A0(v32);
    }
    *(_DWORD *)(a4 + 28) = ((0xBF58476D1CE4E5B9LL
                           * (v33
                            | (((0xBF58476D1CE4E5B9LL
                               * (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)
                                | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32)))
                              ^ ((0xBF58476D1CE4E5B9LL
                                * (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)
                                 | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)) << 32))) >> 31)
                         ^ (484763065 * v33);
    v15 = ((0xBF58476D1CE4E5B9LL
          * (v33
           | (((0xBF58476D1CE4E5B9LL
              * (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)
               | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32)))
             ^ ((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)
                | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)) << 32))) >> 31)
        ^ (484763065 * v33);
  }
  v18 = v13 - 1;
  v19 = 1;
  for ( i = v18 & v15; ; i = v66 & v24 )
  {
    v21 = (__int64 ***)(v14 + 8LL * i);
    v22 = *v21;
    if ( **v21 == v16 && v22[1] == v17 )
      break;
LABEL_12:
    v62 = v6;
    v64 = i;
    v66 = v18;
    v72 = v14;
    v23 = sub_2561100((__int64)v22, &qword_4FEF260);
    v14 = v72;
    v18 = v66;
    v6 = v62;
    if ( v23 )
      goto LABEL_2;
    v24 = v19 + v64;
    v16 = *(__int64 **)a4;
    v17 = *(__int64 **)(a4 + 8);
    ++v19;
  }
  v60 = (_QWORD **)(v14 + 8LL * i);
  v63 = v6;
  v65 = i;
  v69 = v18;
  v74 = v14;
  v53 = sub_254C7C0(*(__int64 **)(a4 + 16), (__int64)v22[2]);
  v14 = v74;
  v18 = v69;
  i = v65;
  v6 = v63;
  v54 = v60;
  if ( !v53 )
  {
    v22 = (__int64 **)*v60;
    goto LABEL_12;
  }
  if ( !byte_4FEF208[0] )
  {
    v55 = sub_2207590((__int64)byte_4FEF208);
    v54 = v60;
    LOBYTE(v6) = v63;
    if ( v55 )
    {
      qword_4FEF220 = -8192;
      unk_4FEF228 = -8192;
      qword_4FEF230 = 0;
      unk_4FEF238 = 0;
      sub_2207640((__int64)byte_4FEF208);
      LOBYTE(v6) = v63;
      v54 = v60;
    }
  }
  *v54 = &qword_4FEF220;
  --*(_DWORD *)(a1 + 184);
  ++*(_DWORD *)(a1 + 188);
LABEL_2:
  v11 = a3 & 1;
  v71 = a6 & (a3 ^ 1) & 1;
  if ( (_BYTE)v6 != 1 || (_BYTE)v11 )
  {
    v25 = *(__int64 **)a4;
    v56 = v6;
    v58 = (a3 ^ 1) & 1;
    v77[1] = *(__int64 **)(a4 + 8);
    v76 = (__int64 *)v77;
    v77[0] = v25;
    v77[2] = 0;
    v77[3] = 0;
    v67 = a1 + 168;
    v26 = sub_2568130(a1 + 168, (__int64 *)&v76);
    v30 = v58;
    v31 = v56;
    if ( !v26 )
    {
      v43 = sub_A777F0(0x20u, *(__int64 **)(a2 + 128));
      v45 = v58;
      v46 = v56;
      if ( v43 )
      {
        v47 = *(_QWORD *)(a4 + 8);
        v48 = *(__int64 **)a4;
        *(_QWORD *)(v43 + 16) = 0;
        *(_DWORD *)(v43 + 28) = 0;
        *(_QWORD *)v43 = v48;
        *(_QWORD *)(v43 + 8) = v47;
      }
      *(_DWORD *)(v43 + 24) = a3;
      v49 = *(unsigned int *)(a1 + 112);
      v50 = *(unsigned int *)(a1 + 116);
      v75 = (__int64 *)v43;
      if ( v49 + 1 > v50 )
      {
        v61 = v43;
        sub_C8D5F0(a1 + 104, (const void *)(a1 + 120), v49 + 1, 8u, v45, v44);
        v49 = *(unsigned int *)(a1 + 112);
        v46 = v56;
        LOBYTE(v45) = (a3 ^ 1) & 1;
        v43 = v61;
      }
      v57 = v46;
      v59 = v45;
      *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v49) = v43;
      ++*(_DWORD *)(a1 + 112);
      v51 = sub_25682F0(v67, (__int64 *)&v75, &v76);
      v30 = v59;
      v31 = v57;
      if ( !v51 )
      {
        v52 = sub_2576D50(v67, (__int64 *)&v75, v76);
        v27 = v75;
        v31 = v57;
        v30 = v59;
        *v52 = (__int64)v75;
      }
    }
    if ( (v31 & a6) != 0 && (_BYTE)v30 )
      goto LABEL_26;
  }
  else
  {
    if ( !v71 )
      return v11;
    v67 = a1 + 168;
LABEL_26:
    v37 = (__int64 *)sub_A777F0(0x20u, *(__int64 **)(a2 + 128));
    if ( v37 )
    {
      v38 = *(_QWORD *)(a4 + 16);
      v39 = *(_QWORD *)(a4 + 8);
      v40 = *(__int64 **)a4;
      v37[3] = 0;
      v37[1] = v39;
      *v37 = (__int64)v40;
      v37[2] = v38;
      if ( !v38 || *(_DWORD *)(v38 + 20) == *(_DWORD *)(v38 + 24) )
        v37[2] = 0;
      else
        v37[2] = sub_256ABE0(*(_QWORD *)(a2 + 208), v38);
    }
    v76 = v37;
    *((_DWORD *)v37 + 6) = 0;
    v41 = *(unsigned int *)(a1 + 112);
    if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 116) )
    {
      sub_C8D5F0(a1 + 104, (const void *)(a1 + 120), v41 + 1, 8u, v35, v36);
      v41 = *(unsigned int *)(a1 + 112);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v41) = v37;
    ++*(_DWORD *)(a1 + 112);
    if ( !sub_25682F0(v67, (__int64 *)&v76, v77) )
    {
      v42 = sub_2576D50(v67, (__int64 *)&v76, v77[0]);
      v27 = v76;
      *v42 = (__int64)v76;
    }
  }
  if ( v71 )
    sub_2519FA0(a2, a1, (__int64)v27, v28, v30, v29);
  return v11;
}
