// Function: sub_2F53AC0
// Address: 0x2f53ac0
//
__int64 __fastcall sub_2F53AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // r12d
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 (*v19)(); // rsi
  __int64 v20; // rax
  unsigned int v21; // eax
  _QWORD **v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rbx
  bool v29; // zf
  _QWORD **v30; // rsi
  _QWORD **v31; // rdx
  _QWORD **v32; // rax
  __int64 v33; // rcx
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // r9
  _QWORD *v38; // rdx
  unsigned int v39; // eax
  unsigned int v40; // r14d
  __int64 v41; // r15
  __int64 v42; // r13
  unsigned __int64 v43; // rax
  __int64 v44; // r12
  unsigned int v45; // edx
  __int64 v46; // rax
  __int64 v47; // r14
  int v48; // r10d
  unsigned __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // r12
  unsigned int v52; // eax
  __int64 v53; // rsi
  __int64 v54; // rax
  unsigned __int64 v55; // r11
  unsigned __int64 v56; // rdx
  __int64 v57; // r12
  unsigned int v58; // ecx
  int v59; // r10d
  unsigned __int64 v60; // r11
  _DWORD *v61; // rax
  unsigned __int64 v62; // rdx
  __int64 *v63; // rax
  __int64 v64; // rax
  unsigned __int64 v65; // r13
  _QWORD *v66; // rcx
  _QWORD *v67; // rdi
  const char *v68; // r12
  __int64 *v69; // rax
  unsigned __int64 v70; // r10
  _DWORD *v71; // rax
  unsigned __int64 v72; // rdx
  int v73; // [rsp+8h] [rbp-168h]
  unsigned __int64 v74; // [rsp+8h] [rbp-168h]
  unsigned int v75; // [rsp+10h] [rbp-160h]
  __int64 v76; // [rsp+10h] [rbp-160h]
  unsigned int v77; // [rsp+10h] [rbp-160h]
  unsigned __int64 v78; // [rsp+18h] [rbp-158h]
  int v79; // [rsp+18h] [rbp-158h]
  unsigned int v80; // [rsp+18h] [rbp-158h]
  const void *v81; // [rsp+20h] [rbp-150h]
  int v82; // [rsp+38h] [rbp-138h]
  __int64 v83; // [rsp+38h] [rbp-138h]
  unsigned __int64 v84[2]; // [rsp+40h] [rbp-130h] BYREF
  _BYTE v85[32]; // [rsp+50h] [rbp-120h] BYREF
  _QWORD v86[2]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v87; // [rsp+80h] [rbp-F0h]
  __int64 v88; // [rsp+88h] [rbp-E8h]
  __int64 v89; // [rsp+90h] [rbp-E0h]
  __int64 v90; // [rsp+98h] [rbp-D8h]
  __int64 v91; // [rsp+A0h] [rbp-D0h]
  __int64 v92; // [rsp+A8h] [rbp-C8h]
  unsigned int v93; // [rsp+B0h] [rbp-C0h]
  char v94; // [rsp+B4h] [rbp-BCh]
  __int64 v95; // [rsp+B8h] [rbp-B8h]
  __int64 v96; // [rsp+C0h] [rbp-B0h]
  char *v97; // [rsp+C8h] [rbp-A8h]
  __int64 v98; // [rsp+D0h] [rbp-A0h]
  int v99; // [rsp+D8h] [rbp-98h]
  char v100; // [rsp+DCh] [rbp-94h]
  char v101; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v102; // [rsp+100h] [rbp-70h]
  char *v103; // [rsp+108h] [rbp-68h]
  __int64 v104; // [rsp+110h] [rbp-60h]
  int v105; // [rsp+118h] [rbp-58h]
  char v106; // [rsp+11Ch] [rbp-54h]
  char v107; // [rsp+120h] [rbp-50h] BYREF

  v82 = *(_DWORD *)(a2 + 112);
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 56LL) + 16LL * (v82 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v10 = 3LL * *(unsigned __int16 *)(*(_QWORD *)v9 + 24LL);
  v11 = *(_QWORD *)(a1 + 48) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v9 + 24LL);
  if ( *(_DWORD *)(a1 + 56) != *(_DWORD *)v11 )
    sub_2F60630(a1 + 48, v9, v10, a4);
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(_QWORD *)(a1 + 32);
  v14 = a1 + 760;
  v15 = *(_QWORD *)(a1 + 768);
  v16 = *(unsigned __int8 *)(v11 + 8);
  v87 = a4;
  v86[1] = a2;
  v86[0] = &unk_4A388F0;
  v17 = *(_QWORD *)(v15 + 32);
  v90 = v12;
  v88 = v17;
  v89 = v13;
  v18 = *(_QWORD *)(v15 + 16);
  v19 = *(__int64 (**)())(*(_QWORD *)v18 + 128LL);
  v20 = 0;
  if ( v19 != sub_2DAC790 )
  {
    v20 = ((__int64 (__fastcall *)(__int64))v19)(v18);
    v17 = v88;
    v14 = a1 + 760;
  }
  v91 = v20;
  v21 = *(_DWORD *)(a4 + 8);
  v92 = v14;
  v93 = v21;
  v97 = &v101;
  v94 = 0;
  v95 = a1 + 400;
  v96 = 0;
  v98 = 4;
  v99 = 0;
  v100 = 1;
  v102 = 0;
  v103 = &v107;
  v104 = 4;
  v105 = 0;
  v106 = 1;
  if ( !*(_BYTE *)(v17 + 36) )
    goto LABEL_27;
  v22 = *(_QWORD ***)(v17 + 16);
  v14 = *(unsigned int *)(v17 + 28);
  v10 = (__int64)&v22[v14];
  if ( v22 == (_QWORD **)v10 )
  {
LABEL_26:
    if ( (unsigned int)v14 >= *(_DWORD *)(v17 + 24) )
    {
LABEL_27:
      sub_C8CC70(v17 + 8, (__int64)v86, v10, v14, v18, a6);
      goto LABEL_10;
    }
    *(_DWORD *)(v17 + 28) = v14 + 1;
    *(_QWORD *)v10 = v86;
    ++*(_QWORD *)(v17 + 8);
  }
  else
  {
    while ( *v22 != v86 )
    {
      if ( (_QWORD **)v10 == ++v22 )
        goto LABEL_26;
    }
  }
LABEL_10:
  sub_2FB3410(*(_QWORD *)(a1 + 1000), v86, (unsigned int)dword_5024248);
  v23 = *(_QWORD *)(a1 + 992);
  v24 = *(_QWORD *)(v23 + 280);
  v25 = v24 + 40LL * *(unsigned int *)(v23 + 288);
  if ( v24 != v25 )
  {
    v26 = *(_QWORD *)(v23 + 280);
    while ( 1 )
    {
      if ( (unsigned __int8)sub_2FB2C20(v23, v26, v16, v24) )
      {
        v27 = v26;
        v26 += 40;
        sub_2FBE000(*(_QWORD *)(a1 + 1000), v27);
        if ( v25 == v26 )
          break;
      }
      else
      {
        v26 += 40;
        if ( v25 == v26 )
          break;
      }
      v23 = *(_QWORD *)(a1 + 992);
    }
  }
  if ( *(_DWORD *)(v87 + 8) != v93 )
  {
    v35 = *(_QWORD *)(a1 + 1000);
    v84[0] = (unsigned __int64)v85;
    v84[1] = 0x800000000LL;
    sub_2FBB760(v35, v84);
    sub_2E01430(
      *(__int64 **)(a1 + 840),
      v82,
      (unsigned int *)(*(_QWORD *)v87 + 4LL * v93),
      *(unsigned int *)(v87 + 8) - (unsigned __int64)v93);
    v38 = (_QWORD *)v87;
    v39 = v93;
    v40 = *(_DWORD *)(v87 + 8) - v93;
    if ( !v40 )
    {
LABEL_48:
      if ( unk_503FCFD )
      {
        v68 = *(const char **)(a1 + 768);
        v69 = (__int64 *)sub_CB72A0();
        sub_2F06090(
          v68,
          *(_QWORD *)(a1 + 32),
          *(_QWORD *)(a1 + 784),
          (__int64)"After splitting live range around basic blocks",
          v69,
          1);
      }
      if ( (_BYTE *)v84[0] != v85 )
        _libc_free(v84[0]);
      goto LABEL_17;
    }
    v41 = 0;
    v83 = v40;
    v81 = (const void *)(a1 + 936);
    while ( 1 )
    {
      v47 = *(_QWORD *)(a1 + 32);
      v48 = *(_DWORD *)(*v38 + 4LL * (v39 + (unsigned int)v41));
      v49 = *(unsigned int *)(v47 + 160);
      v50 = v48 & 0x7FFFFFFF;
      v51 = 8LL * (v48 & 0x7FFFFFFF);
      if ( (v48 & 0x7FFFFFFFu) >= (unsigned int)v49 )
        break;
      v42 = *(_QWORD *)(*(_QWORD *)(v47 + 152) + 8LL * v50);
      if ( !v42 )
        break;
LABEL_31:
      v43 = *(unsigned int *)(a1 + 928);
      v44 = *(_DWORD *)(v42 + 112) & 0x7FFFFFFF;
      v45 = v44 + 1;
      if ( (int)v44 + 1 > (unsigned int)v43 )
      {
        v36 = v45;
        if ( v45 != v43 )
        {
          if ( v45 >= v43 )
          {
            v59 = *(_DWORD *)(a1 + 936);
            v37 = *(unsigned int *)(a1 + 940);
            v60 = v45 - v43;
            if ( v45 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
            {
              v73 = *(_DWORD *)(a1 + 936);
              v75 = *(_DWORD *)(a1 + 940);
              v78 = v45 - v43;
              sub_C8D5F0(a1 + 920, v81, v45, 8u, v45, v37);
              v43 = *(unsigned int *)(a1 + 928);
              v59 = v73;
              v37 = v75;
              v60 = v78;
            }
            v61 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v43);
            v62 = v60;
            do
            {
              if ( v61 )
              {
                *v61 = v59;
                v61[1] = v37;
              }
              v61 += 2;
              --v62;
            }
            while ( v62 );
            *(_DWORD *)(a1 + 928) += v60;
          }
          else
          {
            *(_DWORD *)(a1 + 928) = v45;
          }
        }
      }
      v46 = *(_QWORD *)(a1 + 920);
      if ( *(_DWORD *)(v46 + 8 * v44) || *(_DWORD *)(v84[0] + 4 * v41) )
      {
        if ( ++v41 == v83 )
          goto LABEL_48;
      }
      else
      {
        v56 = *(unsigned int *)(a1 + 928);
        v57 = *(_DWORD *)(v42 + 112) & 0x7FFFFFFF;
        v58 = v57 + 1;
        if ( (int)v57 + 1 > (unsigned int)v56 && v58 != v56 )
        {
          if ( v58 >= v56 )
          {
            v36 = *(unsigned int *)(a1 + 936);
            v37 = *(unsigned int *)(a1 + 940);
            v70 = v58 - v56;
            if ( v58 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
            {
              v74 = v58 - v56;
              v77 = *(_DWORD *)(a1 + 940);
              v80 = *(_DWORD *)(a1 + 936);
              sub_C8D5F0(a1 + 920, v81, v58, 8u, v36, v37);
              v46 = *(_QWORD *)(a1 + 920);
              v56 = *(unsigned int *)(a1 + 928);
              v70 = v74;
              v37 = v77;
              v36 = v80;
            }
            v71 = (_DWORD *)(v46 + 8 * v56);
            v72 = v70;
            do
            {
              if ( v71 )
              {
                *v71 = v36;
                v71[1] = v37;
              }
              v71 += 2;
              --v72;
            }
            while ( v72 );
            *(_DWORD *)(a1 + 928) += v70;
            v46 = *(_QWORD *)(a1 + 920);
          }
          else
          {
            *(_DWORD *)(a1 + 928) = v58;
          }
        }
        *(_DWORD *)(v46 + 8 * v57) = 4;
        if ( ++v41 == v83 )
          goto LABEL_48;
      }
      v38 = (_QWORD *)v87;
      v39 = v93;
    }
    v52 = v50 + 1;
    if ( (unsigned int)v49 < v52 )
    {
      v55 = v52;
      if ( v52 != v49 )
      {
        if ( v52 >= v49 )
        {
          v64 = *(_QWORD *)(v47 + 168);
          v65 = v55 - v49;
          if ( v55 > *(unsigned int *)(v47 + 164) )
          {
            v76 = *(_QWORD *)(v47 + 168);
            v79 = v48;
            sub_C8D5F0(v47 + 152, (const void *)(v47 + 168), v55, 8u, v36, v37);
            v49 = *(unsigned int *)(v47 + 160);
            v64 = v76;
            v48 = v79;
          }
          v53 = *(_QWORD *)(v47 + 152);
          v66 = (_QWORD *)(v53 + 8 * v49);
          v67 = &v66[v65];
          if ( v66 != v67 )
          {
            do
              *v66++ = v64;
            while ( v67 != v66 );
            LODWORD(v49) = *(_DWORD *)(v47 + 160);
            v53 = *(_QWORD *)(v47 + 152);
          }
          *(_DWORD *)(v47 + 160) = v65 + v49;
          goto LABEL_39;
        }
        *(_DWORD *)(v47 + 160) = v52;
      }
    }
    v53 = *(_QWORD *)(v47 + 152);
LABEL_39:
    v54 = sub_2E10F30(v48);
    *(_QWORD *)(v53 + v51) = v54;
    v42 = v54;
    sub_2E11E80((_QWORD *)v47, v54);
    goto LABEL_31;
  }
LABEL_17:
  v28 = v88;
  v29 = *(_BYTE *)(v88 + 36) == 0;
  v86[0] = &unk_4A388F0;
  if ( !v29 )
  {
    v30 = *(_QWORD ***)(v88 + 16);
    v31 = &v30[*(unsigned int *)(v88 + 28)];
    v32 = v30;
    if ( v30 != v31 )
    {
      while ( *v32 != v86 )
      {
        if ( v31 == ++v32 )
          goto LABEL_23;
      }
      v33 = (unsigned int)(*(_DWORD *)(v88 + 28) - 1);
      *(_DWORD *)(v88 + 28) = v33;
      *v32 = v30[v33];
      ++*(_QWORD *)(v28 + 8);
    }
LABEL_23:
    if ( v106 )
      goto LABEL_24;
LABEL_61:
    _libc_free((unsigned __int64)v103);
    if ( v100 )
      return 0;
LABEL_62:
    _libc_free((unsigned __int64)v97);
    return 0;
  }
  v63 = sub_C8CA60(v88 + 8, (__int64)v86);
  if ( !v63 )
    goto LABEL_23;
  *v63 = -2;
  ++*(_DWORD *)(v28 + 32);
  ++*(_QWORD *)(v28 + 8);
  if ( !v106 )
    goto LABEL_61;
LABEL_24:
  if ( !v100 )
    goto LABEL_62;
  return 0;
}
