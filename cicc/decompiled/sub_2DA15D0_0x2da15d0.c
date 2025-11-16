// Function: sub_2DA15D0
// Address: 0x2da15d0
//
__int64 __fastcall sub_2DA15D0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, void *a6)
{
  __int64 v6; // rcx
  unsigned int v7; // eax
  unsigned __int8 **v8; // rdx
  __int64 v9; // r14
  unsigned __int8 *v10; // r15
  unsigned __int8 **v11; // rax
  unsigned int v12; // r12d
  char v14; // dl
  int v15; // eax
  __int64 v16; // rbx
  unsigned __int8 *v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  int v21; // eax
  unsigned __int64 *v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rsi
  unsigned __int64 *v26; // rcx
  _DWORD *v27; // rbx
  __int64 *v28; // rdx
  bool v29; // zf
  unsigned __int8 v30; // al
  int v31; // eax
  __int64 v32; // rdx
  char v33; // cl
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 v36; // rax
  char v37; // dl
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rax
  bool v41; // bl
  unsigned __int8 *v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rcx
  unsigned __int64 *v45; // rcx
  bool v46; // bl
  unsigned __int8 *v47; // rax
  __int64 v48; // rdx
  unsigned __int64 *v49; // rsi
  char v50; // al
  unsigned __int8 *v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // rbx
  int v54; // eax
  unsigned __int8 *v55; // rcx
  unsigned __int8 *v56; // rdx
  unsigned __int8 *v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rdx
  int v60; // eax
  unsigned __int64 *v61; // rsi
  unsigned __int8 *v62; // rsi
  unsigned __int64 *v63; // rcx
  __int64 *v64; // rdx
  __int64 *v65; // rax
  signed __int64 v66; // rdx
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // rsi
  bool v69; // cf
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rsi
  __int64 v72; // rax
  char *v73; // r10
  char *v74; // rax
  __int64 v75; // rbx
  unsigned __int64 v76; // rbx
  unsigned __int64 v77; // rbx
  unsigned __int64 v78; // rdx
  char *v79; // rax
  unsigned __int64 v80; // rbx
  size_t n; // [rsp+8h] [rbp-118h]
  void *src; // [rsp+10h] [rbp-110h]
  void *srca; // [rsp+10h] [rbp-110h]
  __int64 v84; // [rsp+18h] [rbp-108h]
  char *v85; // [rsp+18h] [rbp-108h]
  unsigned __int64 v86; // [rsp+20h] [rbp-100h]
  unsigned __int8 **v90; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v91; // [rsp+58h] [rbp-C8h]
  _QWORD v92[6]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v93; // [rsp+90h] [rbp-90h] BYREF
  unsigned __int8 **v94; // [rsp+98h] [rbp-88h]
  __int64 v95; // [rsp+A0h] [rbp-80h]
  int v96; // [rsp+A8h] [rbp-78h]
  unsigned __int8 v97; // [rsp+ACh] [rbp-74h]
  char v98; // [rsp+B0h] [rbp-70h] BYREF

  v90 = (unsigned __int8 **)v92;
  v92[0] = (unsigned __int64)a2 | 4;
  v93 = 0;
  v95 = 8;
  v96 = 0;
  v97 = 1;
  v91 = 0x600000001LL;
  v6 = 1;
  v94 = (unsigned __int8 **)&v98;
  v7 = 1;
  while ( 2 )
  {
    if ( !v7 )
    {
LABEL_9:
      v12 = 1;
      goto LABEL_10;
    }
    while ( 1 )
    {
      v8 = v90;
      v9 = (__int64)v90[v7 - 1];
      LODWORD(v91) = v7 - 1;
      v10 = (unsigned __int8 *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_BYTE)v6 )
      {
        v11 = v94;
        v8 = &v94[HIDWORD(v95)];
        if ( v94 != v8 )
        {
          while ( v10 != *v11 )
          {
            if ( v8 == ++v11 )
              goto LABEL_71;
          }
          goto LABEL_8;
        }
LABEL_71:
        if ( HIDWORD(v95) < (unsigned int)v95 )
          break;
      }
      sub_C8CC70((__int64)&v93, v9 & 0xFFFFFFFFFFFFFFF8LL, (__int64)v8, v6, a5, (__int64)a6);
      v6 = v97;
      if ( v14 )
        goto LABEL_16;
LABEL_8:
      v7 = v91;
      if ( !(_DWORD)v91 )
        goto LABEL_9;
    }
    ++HIDWORD(v95);
    *v8 = v10;
    ++v93;
LABEL_16:
    v15 = *v10;
    v16 = (v9 >> 2) & 1;
    if ( (unsigned __int8)v15 <= 0x1Cu )
      goto LABEL_73;
    if ( a2 == v10 )
      goto LABEL_20;
    if ( (unsigned int)sub_BD3960(v9 & 0xFFFFFFFFFFFFFFF8LL) > 1 )
    {
LABEL_73:
      v52 = sub_22077B0(0x20u);
      *(_QWORD *)(v52 + 16) = v10;
      *(_BYTE *)(v52 + 24) = v16;
      sub_2208C80((_QWORD *)v52, a4);
      ++*(_QWORD *)(a4 + 16);
LABEL_34:
      v7 = v91;
      v6 = v97;
      continue;
    }
    break;
  }
  v15 = *v10;
LABEL_20:
  switch ( v15 )
  {
    case ')':
      v41 = !((v9 >> 2) & 1);
      if ( (v10[7] & 0x40) != 0 )
        goto LABEL_56;
      goto LABEL_79;
    case '*':
    case '+':
      if ( (v10[7] & 0x40) != 0 )
        v17 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v17 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v18 = (unsigned int)v91;
      v19 = HIDWORD(v91);
      v20 = *((_QWORD *)v17 + 4);
      v21 = v91;
      if ( (unsigned int)v91 < (unsigned __int64)HIDWORD(v91) )
      {
        v22 = (unsigned __int64 *)&v90[(unsigned int)v91];
        if ( v22 )
        {
          *v22 = v9 & 4 | v20 & 0xFFFFFFFFFFFFFFFBLL;
          v21 = v91;
          v19 = HIDWORD(v91);
        }
        goto LABEL_26;
      }
      v77 = v20 & 0xFFFFFFFFFFFFFFFBLL | v9 & 4;
      v78 = (unsigned int)v91 + 1LL;
      if ( HIDWORD(v91) >= v78 )
        goto LABEL_138;
      goto LABEL_140;
    case ',':
      v41 = !((v9 >> 2) & 1);
      if ( (unsigned __int8)sub_2DA10D0(v9 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v50 = v10[7] & 0x40;
        if ( *v10 == 41 )
        {
          if ( v50 )
LABEL_56:
            v42 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
          else
LABEL_79:
            v42 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
          v43 = *(_QWORD *)v42;
        }
        else
        {
          if ( v50 )
            v51 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
          else
            v51 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
          v43 = *((_QWORD *)v51 + 4);
        }
        v44 = (unsigned int)v91;
        v23 = v91;
        if ( (unsigned int)v91 < (unsigned __int64)HIDWORD(v91) )
        {
          v45 = (unsigned __int64 *)&v90[(unsigned int)v91];
          if ( v45 )
          {
            *v45 = (4LL * v41) | v43 & 0xFFFFFFFFFFFFFFFBLL;
            v23 = v91;
          }
          goto LABEL_31;
        }
        v76 = v43 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v41);
        if ( HIDWORD(v91) < (unsigned __int64)(unsigned int)v91 + 1 )
        {
          sub_C8D5F0((__int64)&v90, v92, (unsigned int)v91 + 1LL, 8u, a5, (__int64)a6);
          v44 = (unsigned int)v91;
        }
        v90[v44] = (unsigned __int8 *)v76;
        LODWORD(v91) = v91 + 1;
        goto LABEL_32;
      }
      if ( (v10[7] & 0x40) != 0 )
        v57 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v57 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v58 = (unsigned int)v91;
      v19 = HIDWORD(v91);
      v59 = *((_QWORD *)v57 + 4);
      v60 = v91;
      if ( (unsigned int)v91 >= (unsigned __int64)HIDWORD(v91) )
      {
        v80 = v59 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v41);
        if ( HIDWORD(v91) < (unsigned __int64)(unsigned int)v91 + 1 )
        {
          sub_C8D5F0((__int64)&v90, v92, (unsigned int)v91 + 1LL, 8u, a5, (__int64)a6);
          v58 = (unsigned int)v91;
        }
        v90[v58] = (unsigned __int8 *)v80;
        v19 = HIDWORD(v91);
        v23 = v91 + 1;
        LODWORD(v91) = v91 + 1;
      }
      else
      {
        v61 = (unsigned __int64 *)&v90[(unsigned int)v91];
        if ( v61 )
        {
          *v61 = v59 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v41);
          v60 = v91;
          v19 = HIDWORD(v91);
        }
        v23 = v60 + 1;
        LODWORD(v91) = v23;
      }
      if ( (v10[7] & 0x40) != 0 )
        v62 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v62 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v24 = *(_QWORD *)v62;
      v25 = v23;
      if ( v23 < v19 )
      {
        v63 = (unsigned __int64 *)&v90[v23];
        if ( v63 )
        {
          *v63 = v24 & 0xFFFFFFFFFFFFFFFBLL | v9 & 4;
          v23 = v91;
        }
        goto LABEL_31;
      }
      goto LABEL_75;
    case '-':
      v46 = !((v9 >> 2) & 1);
      if ( (v10[7] & 0x40) != 0 )
        v47 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v47 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v18 = (unsigned int)v91;
      v19 = HIDWORD(v91);
      v48 = *((_QWORD *)v47 + 4);
      v21 = v91;
      if ( (unsigned int)v91 >= (unsigned __int64)HIDWORD(v91) )
      {
        v77 = v48 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v46);
        v78 = (unsigned int)v91 + 1LL;
        if ( HIDWORD(v91) < v78 )
        {
LABEL_140:
          sub_C8D5F0((__int64)&v90, v92, v78, 8u, a5, (__int64)a6);
          v18 = (unsigned int)v91;
        }
LABEL_138:
        v90[v18] = (unsigned __int8 *)v77;
        v19 = HIDWORD(v91);
        v23 = v91 + 1;
        LODWORD(v91) = v91 + 1;
      }
      else
      {
        v49 = (unsigned __int64 *)&v90[(unsigned int)v91];
        if ( v49 )
        {
          *v49 = (4LL * v46) | v48 & 0xFFFFFFFFFFFFFFFBLL;
          v21 = v91;
          v19 = HIDWORD(v91);
        }
LABEL_26:
        v23 = v21 + 1;
        LODWORD(v91) = v23;
      }
      if ( (v10[7] & 0x40) != 0 )
      {
        v24 = **((_QWORD **)v10 - 1);
        v25 = v23;
        if ( v23 >= v19 )
          goto LABEL_75;
      }
      else
      {
        v24 = *(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
        v25 = v23;
        if ( v23 >= v19 )
        {
LABEL_75:
          v53 = v24 & 0xFFFFFFFFFFFFFFFBLL | v9 & 4;
          if ( v25 + 1 > v19 )
          {
            sub_C8D5F0((__int64)&v90, v92, v25 + 1, 8u, a5, (__int64)a6);
            v25 = (unsigned int)v91;
          }
          v90[v25] = (unsigned __int8 *)v53;
          LODWORD(v91) = v91 + 1;
          goto LABEL_32;
        }
      }
      v26 = (unsigned __int64 *)&v90[v25];
      if ( v26 )
      {
        *v26 = v9 & 4 | v24 & 0xFFFFFFFFFFFFFFFBLL;
        v23 = v91;
      }
LABEL_31:
      LODWORD(v91) = v23 + 1;
      goto LABEL_32;
    case '.':
    case '/':
      if ( (v10[7] & 0x40) != 0 )
        v28 = (__int64 *)*((_QWORD *)v10 - 1);
      else
        v28 = (__int64 *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v29 = (unsigned __int8)sub_2DA10D0(*v28) == 0;
      v30 = v10[7];
      if ( !v29 )
      {
        v31 = v30 & 0x40;
        if ( v31 )
        {
          v32 = **((_QWORD **)v10 - 1);
          v33 = *(_BYTE *)(v32 + 7) & 0x40;
          if ( *(_BYTE *)v32 != 41 )
            goto LABEL_40;
        }
        else
        {
          v32 = *(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
          v33 = *(_BYTE *)(v32 + 7) & 0x40;
          if ( *(_BYTE *)v32 != 41 )
          {
LABEL_40:
            if ( v33 )
              v34 = *(_QWORD *)(v32 - 8);
            else
              v34 = v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF);
            v35 = *(_QWORD *)(v34 + 32);
            goto LABEL_43;
          }
        }
        if ( v33 )
          v64 = *(__int64 **)(v32 - 8);
        else
          v64 = (__int64 *)(v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF));
        v35 = *v64;
LABEL_43:
        LOBYTE(v16) = v16 ^ 1;
        if ( (_BYTE)v31 )
          goto LABEL_44;
LABEL_84:
        if ( !(unsigned __int8)sub_2DA10D0(*(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF) + 32]) )
          goto LABEL_85;
LABEL_45:
        if ( (v10[7] & 0x40) != 0 )
        {
          v36 = *(_QWORD *)(*((_QWORD *)v10 - 1) + 32LL);
          v37 = *(_BYTE *)(v36 + 7) & 0x40;
          if ( *(_BYTE *)v36 != 41 )
          {
LABEL_47:
            if ( v37 )
              v38 = *(_QWORD *)(v36 - 8);
            else
              v38 = v36 - 32LL * (*(_DWORD *)(v36 + 4) & 0x7FFFFFF);
            v39 = *(_QWORD *)(v38 + 32);
LABEL_50:
            LOBYTE(v16) = v16 ^ 1;
            goto LABEL_51;
          }
        }
        else
        {
          v36 = *(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF) + 32];
          v37 = *(_BYTE *)(v36 + 7) & 0x40;
          if ( *(_BYTE *)v36 != 41 )
            goto LABEL_47;
        }
        if ( v37 )
          v65 = *(__int64 **)(v36 - 8);
        else
          v65 = (__int64 *)(v36 - 32LL * (*(_DWORD *)(v36 + 4) & 0x7FFFFFF));
        v39 = *v65;
        goto LABEL_50;
      }
      v54 = v30 & 0x40;
      if ( v54 )
        v55 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v55 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v35 = *(_QWORD *)v55;
      if ( !(_BYTE)v54 )
        goto LABEL_84;
LABEL_44:
      if ( (unsigned __int8)sub_2DA10D0(*(_QWORD *)(*((_QWORD *)v10 - 1) + 32LL)) )
        goto LABEL_45;
LABEL_85:
      if ( (v10[7] & 0x40) != 0 )
        v56 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v56 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v39 = *((_QWORD *)v56 + 4);
LABEL_51:
      v40 = *(_QWORD *)(a3 + 8);
      if ( v40 == *(_QWORD *)(a3 + 16) )
      {
        a6 = *(void **)a3;
        v66 = v40 - *(_QWORD *)a3;
        v67 = 0xAAAAAAAAAAAAAAABLL * (v66 >> 3);
        if ( v67 == 0x555555555555555LL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v68 = 1;
        if ( v67 )
          v68 = 0xAAAAAAAAAAAAAAABLL * (v66 >> 3);
        v69 = __CFADD__(v68, v67);
        v70 = v68 - 0x5555555555555555LL * (v66 >> 3);
        if ( v69 )
        {
          v71 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v70 )
          {
            v86 = 0;
            v73 = 0;
LABEL_122:
            v74 = &v73[v66];
            if ( &v73[v66] )
            {
              *(_QWORD *)v74 = v35;
              *((_QWORD *)v74 + 1) = v39;
              v74[16] = v16;
            }
            v75 = (__int64)&v73[v66 + 24];
            if ( v66 > 0 )
            {
              srca = a6;
              v79 = (char *)memmove(v73, a6, v66);
              a6 = srca;
              v73 = v79;
            }
            else if ( !a6 )
            {
LABEL_126:
              *(_QWORD *)a3 = v73;
              *(_QWORD *)(a3 + 8) = v75;
              *(_QWORD *)(a3 + 16) = v86;
              goto LABEL_32;
            }
            v85 = v73;
            j_j___libc_free_0((unsigned __int64)a6);
            v73 = v85;
            goto LABEL_126;
          }
          if ( v70 > 0x555555555555555LL )
            v70 = 0x555555555555555LL;
          v71 = 24 * v70;
        }
        n = v66;
        src = *(void **)a3;
        v84 = v39;
        v72 = sub_22077B0(v71);
        v39 = v84;
        v73 = (char *)v72;
        a6 = src;
        v66 = n;
        v86 = v71 + v72;
        goto LABEL_122;
      }
      if ( v40 )
      {
        *(_QWORD *)v40 = v35;
        *(_QWORD *)(v40 + 8) = v39;
        *(_BYTE *)(v40 + 16) = v16;
        v40 = *(_QWORD *)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v40 + 24;
LABEL_32:
      if ( !*(_BYTE *)(*(_QWORD *)a1 + 4LL) )
        goto LABEL_34;
      v27 = *(_DWORD **)a1;
      if ( (unsigned int)sub_B45210((__int64)v10) == *v27 )
        goto LABEL_34;
      LOBYTE(v6) = v97;
      v12 = 0;
LABEL_10:
      if ( !(_BYTE)v6 )
        _libc_free((unsigned __int64)v94);
      if ( v90 != v92 )
        _libc_free((unsigned __int64)v90);
      return v12;
    default:
      goto LABEL_73;
  }
}
