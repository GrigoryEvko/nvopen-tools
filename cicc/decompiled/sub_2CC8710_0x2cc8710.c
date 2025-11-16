// Function: sub_2CC8710
// Address: 0x2cc8710
//
_QWORD *__fastcall sub_2CC8710(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v9; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rax
  void *v19; // r9
  bool v20; // zf
  __int64 v21; // r15
  __int64 v22; // r8
  int v23; // r14d
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  unsigned int v26; // eax
  __int64 v27; // r11
  __int64 v28; // rax
  _QWORD *v29; // r14
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // r9
  int v33; // esi
  unsigned int v34; // edx
  _QWORD *v35; // rax
  __int64 v36; // r11
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 *v39; // rdi
  __int64 v40; // r12
  char v41; // cl
  unsigned int v42; // esi
  __int64 v43; // rcx
  const char *v44; // r8
  __int64 v45; // rdi
  __int64 v46; // rdx
  unsigned int v47; // eax
  _QWORD *v48; // r10
  int v49; // edx
  unsigned int v50; // edi
  __int64 v51; // rsi
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rsi
  int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rcx
  int v59; // r9d
  _QWORD *v60; // rdi
  int v61; // edx
  int v62; // edx
  int v63; // r9d
  int v64; // [rsp+10h] [rbp-170h]
  __int64 v65; // [rsp+10h] [rbp-170h]
  __int64 v66; // [rsp+18h] [rbp-168h]
  void *v67; // [rsp+18h] [rbp-168h]
  __int64 v68; // [rsp+20h] [rbp-160h]
  __int64 v69; // [rsp+20h] [rbp-160h]
  _QWORD v70[2]; // [rsp+30h] [rbp-150h] BYREF
  _QWORD v71[11]; // [rsp+40h] [rbp-140h] BYREF
  char *v72; // [rsp+98h] [rbp-E8h]
  __int64 v73; // [rsp+A0h] [rbp-E0h]
  int v74; // [rsp+A8h] [rbp-D8h]
  char v75; // [rsp+ACh] [rbp-D4h]
  char v76; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v77; // [rsp+D0h] [rbp-B0h]
  __int64 v78; // [rsp+D8h] [rbp-A8h]
  __int64 v79; // [rsp+E0h] [rbp-A0h]
  __int64 v80; // [rsp+E8h] [rbp-98h]
  __int64 v81; // [rsp+F0h] [rbp-90h]
  __int64 v82; // [rsp+F8h] [rbp-88h]
  __int64 v83; // [rsp+100h] [rbp-80h]
  __int64 v84; // [rsp+108h] [rbp-78h]
  void *src; // [rsp+110h] [rbp-70h]
  __int64 v86; // [rsp+118h] [rbp-68h]
  _BYTE v87[32]; // [rsp+120h] [rbp-60h] BYREF
  __int16 v88; // [rsp+140h] [rbp-40h]

  v7 = a5;
  v9 = a3;
  v11 = *(_QWORD *)(a5 + 16);
  v12 = *(_QWORD *)(a5 + 24);
  v71[1] = 0;
  memset(&v71[4], 0, 56);
  v71[3] = v11;
  v72 = &v76;
  v71[2] = v12;
  v73 = 4;
  v74 = 0;
  v75 = 1;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  src = v87;
  v86 = 0x400000000LL;
  v88 = 0;
  v71[0] = a3;
  if ( (unsigned __int8)sub_D4B3D0(a3) && (unsigned __int8)sub_2CC5900(v71) )
  {
    if ( (_BYTE)v88 )
    {
      v43 = 14;
      v44 = "<unnamed loop>";
      v45 = **(_QWORD **)(v9 + 32);
      if ( v45 && (*(_BYTE *)(v45 + 7) & 0x10) != 0 )
      {
        v44 = sub_BD5D20(v45);
        v43 = v46;
      }
      sub_22D0060(*(_QWORD *)(a6 + 8), v9, (__int64)v44, v43);
      if ( v9 == *(_QWORD *)(a6 + 16) )
        *(_BYTE *)(a6 + 24) = 1;
    }
    if ( (_DWORD)v86 )
    {
      v19 = src;
      v20 = *(_BYTE *)(a6 + 25) == 0;
      v70[1] = (unsigned int)v86;
      v21 = *(_QWORD *)a6;
      v70[0] = src;
      if ( v20 )
      {
        sub_F76FB0(v70, v21, v13, v14, v15, (__int64)src);
        goto LABEL_5;
      }
      v22 = 8LL * (unsigned int)v86;
      v23 = v86;
      v24 = *(unsigned int *)(v21 + 88) + (unsigned __int64)(unsigned int)v86;
      v68 = *(unsigned int *)(v21 + 88);
      v25 = v68;
      if ( v24 > *(unsigned int *)(v21 + 92) )
      {
        v65 = 8LL * (unsigned int)v86;
        v67 = src;
        sub_C8D5F0(v21 + 80, (const void *)(v21 + 96), v24, 8u, v22, (__int64)src);
        v25 = *(unsigned int *)(v21 + 88);
        v22 = v65;
        v19 = v67;
      }
      memcpy((void *)(*(_QWORD *)(v21 + 80) + 8 * v25), v19, v22);
      v26 = v23 + *(_DWORD *)(v21 + 88);
      *(_DWORD *)(v21 + 88) = v26;
      v27 = v26 - 1LL;
      v28 = v68;
      if ( v68 <= v27 )
      {
        v69 = v9;
        v29 = a1;
        v30 = v27;
        v66 = v7;
        v31 = v28;
        while ( 1 )
        {
          v38 = *(_QWORD *)(v21 + 80);
          v39 = (__int64 *)(v38 + 8 * v30);
          v40 = *v39;
          v41 = *(_BYTE *)(v21 + 8) & 1;
          if ( v41 )
          {
            v32 = v21 + 16;
            v33 = 3;
          }
          else
          {
            v42 = *(_DWORD *)(v21 + 24);
            v32 = *(_QWORD *)(v21 + 16);
            if ( !v42 )
            {
              v47 = *(_DWORD *)(v21 + 8);
              ++*(_QWORD *)v21;
              v48 = 0;
              v49 = (v47 >> 1) + 1;
              goto LABEL_32;
            }
            v33 = v42 - 1;
          }
          v34 = v33 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v35 = (_QWORD *)(v32 + 16LL * v34);
          v36 = *v35;
          if ( v40 != *v35 )
            break;
LABEL_18:
          v37 = v35[1];
          if ( v31 > v37 )
          {
            *(_QWORD *)(v38 + 8 * v37) = 0;
            v35[1] = v30;
          }
          else
          {
            *v39 = 0;
          }
LABEL_20:
          if ( v31 > --v30 )
          {
            v9 = v69;
            v7 = v66;
            a1 = v29;
            goto LABEL_5;
          }
        }
        v64 = 1;
        v48 = 0;
        while ( v36 != -4096 )
        {
          if ( v36 == -8192 && !v48 )
            v48 = v35;
          v34 = v33 & (v64 + v34);
          v35 = (_QWORD *)(v32 + 16LL * v34);
          v36 = *v35;
          if ( v40 == *v35 )
            goto LABEL_18;
          ++v64;
        }
        if ( !v48 )
          v48 = v35;
        v47 = *(_DWORD *)(v21 + 8);
        ++*(_QWORD *)v21;
        v49 = (v47 >> 1) + 1;
        if ( v41 )
        {
          v50 = 12;
          v42 = 4;
        }
        else
        {
          v42 = *(_DWORD *)(v21 + 24);
LABEL_32:
          v50 = 3 * v42;
        }
        if ( v50 <= 4 * v49 )
        {
          sub_F76580(v21, 2 * v42);
          if ( (*(_BYTE *)(v21 + 8) & 1) != 0 )
          {
            v51 = v21 + 16;
            v52 = 3;
          }
          else
          {
            v61 = *(_DWORD *)(v21 + 24);
            v51 = *(_QWORD *)(v21 + 16);
            if ( !v61 )
              goto LABEL_79;
            v52 = v61 - 1;
          }
          LODWORD(v53) = v52 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v48 = (_QWORD *)(v51 + 16LL * (unsigned int)v53);
          v54 = *v48;
          if ( *v48 == v40 )
            goto LABEL_49;
          v63 = 1;
          v60 = 0;
          while ( v54 != -4096 )
          {
            if ( !v60 && v54 == -8192 )
              v60 = v48;
            v53 = v52 & (unsigned int)(v53 + v63);
            v48 = (_QWORD *)(v51 + 16 * v53);
            v54 = *v48;
            if ( v40 == *v48 )
              goto LABEL_49;
            ++v63;
          }
        }
        else
        {
          if ( v42 - *(_DWORD *)(v21 + 12) - v49 > v42 >> 3 )
          {
LABEL_35:
            *(_DWORD *)(v21 + 8) = (2 * (v47 >> 1) + 2) | v47 & 1;
            if ( *v48 != -4096 )
              --*(_DWORD *)(v21 + 12);
            *v48 = v40;
            v48[1] = v30;
            goto LABEL_20;
          }
          sub_F76580(v21, v42);
          if ( (*(_BYTE *)(v21 + 8) & 1) != 0 )
          {
            v55 = v21 + 16;
            v56 = 3;
          }
          else
          {
            v62 = *(_DWORD *)(v21 + 24);
            v55 = *(_QWORD *)(v21 + 16);
            if ( !v62 )
            {
LABEL_79:
              *(_DWORD *)(v21 + 8) = (2 * (*(_DWORD *)(v21 + 8) >> 1) + 2) | *(_DWORD *)(v21 + 8) & 1;
              BUG();
            }
            v56 = v62 - 1;
          }
          LODWORD(v57) = v56 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v48 = (_QWORD *)(v55 + 16LL * (unsigned int)v57);
          v58 = *v48;
          if ( v40 == *v48 )
          {
LABEL_49:
            v47 = *(_DWORD *)(v21 + 8);
            goto LABEL_35;
          }
          v59 = 1;
          v60 = 0;
          while ( v58 != -4096 )
          {
            if ( v58 == -8192 && !v60 )
              v60 = v48;
            v57 = v56 & (unsigned int)(v57 + v59);
            v48 = (_QWORD *)(v55 + 16 * v57);
            v58 = *v48;
            if ( v40 == *v48 )
              goto LABEL_49;
            ++v59;
          }
        }
        if ( v60 )
          v48 = v60;
        goto LABEL_49;
      }
    }
LABEL_5:
    v16 = *(_QWORD *)(v7 + 16);
    v17 = *(_QWORD *)(**(_QWORD **)(v9 + 32) + 72LL);
    *(_QWORD *)(v16 + 104) = v17;
    *(_DWORD *)(v16 + 120) = *(_DWORD *)(v17 + 92);
    sub_B1F440(v16);
    memset(a1, 0, 0x60u);
    *((_BYTE *)a1 + 28) = 1;
    a1[1] = a1 + 4;
    *((_DWORD *)a1 + 4) = 2;
    a1[7] = a1 + 10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    *((_BYTE *)a1 + 76) = 1;
    a1[1] = a1 + 4;
    a1[7] = a1 + 10;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  if ( src != v87 )
    _libc_free((unsigned __int64)src);
  if ( !v75 )
    _libc_free((unsigned __int64)v72);
  return a1;
}
