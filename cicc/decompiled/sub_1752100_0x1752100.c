// Function: sub_1752100
// Address: 0x1752100
//
void __fastcall sub_1752100(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // r10
  __int64 v12; // rcx
  __int64 v13; // r15
  unsigned int v14; // eax
  __int64 v15; // rbx
  _BYTE *v16; // r11
  size_t v17; // r8
  _QWORD *v18; // rax
  _BYTE *v19; // rdi
  __int64 v20; // rdx
  char *v21; // r8
  char *v22; // r15
  __int64 v23; // rax
  char *v24; // rcx
  unsigned __int64 v25; // rax
  size_t v26; // rdx
  char *v27; // r9
  size_t v28; // rdx
  unsigned __int64 v29; // rsi
  bool v30; // cf
  unsigned __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  char *v34; // r10
  char *v35; // rax
  char *v36; // rdx
  char *v37; // rax
  char *v38; // rsi
  char *v39; // rcx
  __int64 v40; // r15
  char *v41; // rax
  char *v42; // r15
  __int64 v43; // rax
  _QWORD *v44; // rdi
  size_t v45; // rdx
  _BYTE *v46; // rdi
  char *v47; // [rsp-98h] [rbp-98h]
  size_t v48; // [rsp-98h] [rbp-98h]
  char *v49; // [rsp-90h] [rbp-90h]
  unsigned int v50; // [rsp-88h] [rbp-88h]
  char *v51; // [rsp-88h] [rbp-88h]
  size_t v52; // [rsp-88h] [rbp-88h]
  char *v53; // [rsp-88h] [rbp-88h]
  unsigned int v54; // [rsp-88h] [rbp-88h]
  int v55; // [rsp-88h] [rbp-88h]
  _BYTE *v56; // [rsp-88h] [rbp-88h]
  unsigned int v57; // [rsp-80h] [rbp-80h]
  char *v58; // [rsp-80h] [rbp-80h]
  char *v59; // [rsp-80h] [rbp-80h]
  char *v60; // [rsp-80h] [rbp-80h]
  char *v61; // [rsp-80h] [rbp-80h]
  __int64 v62; // [rsp-80h] [rbp-80h]
  __int64 v63; // [rsp-80h] [rbp-80h]
  unsigned int v64; // [rsp-80h] [rbp-80h]
  __int64 v65; // [rsp-78h] [rbp-78h]
  unsigned int v66; // [rsp-70h] [rbp-70h]
  __int64 v67; // [rsp-70h] [rbp-70h]
  size_t v68; // [rsp-60h] [rbp-60h] BYREF
  _QWORD *v69; // [rsp-58h] [rbp-58h] BYREF
  size_t v70; // [rsp-50h] [rbp-50h]
  _QWORD v71[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(char *)(a1 + 23) < 0 )
  {
    v4 = sub_1648A40(a1);
    v6 = v4 + v5;
    if ( *(char *)(a1 + 23) >= 0 )
      v7 = v6 >> 4;
    else
      LODWORD(v7) = (v6 - sub_1648A40(a1)) >> 4;
    v8 = 0;
    v65 = 16LL * (unsigned int)v7;
    if ( (_DWORD)v7 )
    {
      do
      {
        v9 = 0;
        if ( *(char *)(a1 + 23) < 0 )
          v9 = sub_1648A40(a1);
        v10 = (__int64 *)(v8 + v9);
        v11 = *v10;
        v12 = *((unsigned int *)v10 + 2);
        v13 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
        v66 = *((_DWORD *)v10 + 3);
        v14 = *(_DWORD *)(a2 + 8);
        if ( v14 >= *(_DWORD *)(a2 + 12) )
        {
          v54 = v12;
          v62 = v11;
          sub_1740340(a2, 0);
          v14 = *(_DWORD *)(a2 + 8);
          v12 = v54;
          v11 = v62;
        }
        v15 = *(_QWORD *)a2 + 56LL * v14;
        if ( !v15 )
          goto LABEL_26;
        v16 = (_BYTE *)(v15 + 16);
        *(_BYTE *)(v15 + 16) = 0;
        *(_QWORD *)v15 = v15 + 16;
        *(_QWORD *)(v15 + 8) = 0;
        *(_QWORD *)(v15 + 32) = 0;
        *(_QWORD *)(v15 + 40) = 0;
        *(_QWORD *)(v15 + 48) = 0;
        v17 = *(_QWORD *)v11;
        v69 = v71;
        v68 = v17;
        if ( v17 > 0xF )
        {
          v48 = v17;
          v55 = v12;
          v63 = v11;
          v43 = sub_22409D0(&v69, &v68, 0);
          v11 = v63;
          LODWORD(v12) = v55;
          v69 = (_QWORD *)v43;
          v44 = (_QWORD *)v43;
          v16 = (_BYTE *)(v15 + 16);
          v17 = v48;
          v71[0] = v68;
        }
        else
        {
          if ( v17 == 1 )
          {
            LOBYTE(v71[0]) = *(_BYTE *)(v11 + 16);
            v18 = v71;
            goto LABEL_13;
          }
          if ( !v17 )
          {
            v18 = v71;
            goto LABEL_13;
          }
          v44 = v71;
        }
        v56 = v16;
        v64 = v12;
        memcpy(v44, (const void *)(v11 + 16), v17);
        v17 = v68;
        v18 = v69;
        v12 = v64;
        v16 = v56;
LABEL_13:
        v70 = v17;
        *((_BYTE *)v18 + v17) = 0;
        v19 = *(_BYTE **)v15;
        if ( v69 == v71 )
        {
          v26 = v70;
          if ( !v70 )
            goto LABEL_31;
          if ( v70 != 1 )
          {
            v50 = v12;
            memcpy(v19, v71, v70);
            v26 = v70;
            v19 = *(_BYTE **)v15;
            v12 = v50;
LABEL_31:
            *(_QWORD *)(v15 + 8) = v26;
            v19[v26] = 0;
            v19 = v69;
            goto LABEL_17;
          }
          *v19 = v71[0];
          v45 = v70;
          v46 = *(_BYTE **)v15;
          *(_QWORD *)(v15 + 8) = v70;
          v46[v45] = 0;
          v19 = v69;
        }
        else
        {
          if ( v16 == v19 )
          {
            *(_QWORD *)v15 = v69;
            *(_QWORD *)(v15 + 8) = v70;
            *(_QWORD *)(v15 + 16) = v71[0];
          }
          else
          {
            *(_QWORD *)v15 = v69;
            v20 = *(_QWORD *)(v15 + 16);
            *(_QWORD *)(v15 + 8) = v70;
            *(_QWORD *)(v15 + 16) = v71[0];
            if ( v19 )
            {
              v69 = v19;
              v71[0] = v20;
              goto LABEL_17;
            }
          }
          v69 = v71;
          v19 = v71;
        }
LABEL_17:
        v70 = 0;
        *v19 = 0;
        if ( v69 != v71 )
        {
          v57 = v12;
          j_j___libc_free_0(v69, v71[0] + 1LL);
          v12 = v57;
        }
        v21 = *(char **)(v15 + 40);
        v22 = (char *)(a1 + 24 * v12 - 24 * v13);
        v23 = 24LL * v66 - 24 * v12;
        v24 = &v22[v23];
        if ( v22 == &v22[v23] )
          goto LABEL_25;
        v25 = 0xAAAAAAAAAAAAAAABLL * (v23 >> 3);
        if ( v25 > (__int64)(*(_QWORD *)(v15 + 48) - (_QWORD)v21) >> 3 )
        {
          v27 = *(char **)(v15 + 32);
          v28 = v21 - v27;
          v29 = (v21 - v27) >> 3;
          if ( v25 > 0xFFFFFFFFFFFFFFFLL - v29 )
            sub_4262D8((__int64)"vector::_M_range_insert");
          if ( v25 < v29 )
            v25 = (v21 - v27) >> 3;
          v30 = __CFADD__(v29, v25);
          v31 = v29 + v25;
          if ( !v30 )
          {
            if ( v31 )
            {
              if ( v31 > 0xFFFFFFFFFFFFFFFLL )
                v31 = 0xFFFFFFFFFFFFFFFLL;
              v32 = 8 * v31;
LABEL_40:
              v51 = *(char **)(v15 + 40);
              v58 = v24;
              v33 = sub_22077B0(v32);
              v21 = v51;
              v34 = (char *)v33;
              v27 = *(char **)(v15 + 32);
              v24 = v58;
              v67 = v32 + v33;
              v28 = v51 - v27;
            }
            else
            {
              v67 = 0;
              v34 = 0;
            }
            if ( v21 != v27 )
            {
              v52 = v28;
              v47 = v21;
              v49 = v24;
              v59 = v27;
              v35 = (char *)memmove(v34, v27, v28);
              v21 = v47;
              v24 = v49;
              v28 = v52;
              v27 = v59;
              v34 = v35;
            }
            v36 = &v34[v28];
            v37 = v22;
            v38 = v36;
            do
            {
              if ( v38 )
                *(_QWORD *)v38 = *(_QWORD *)v37;
              v37 += 24;
              v38 += 8;
            }
            while ( v24 != v37 );
            v39 = &v36[0x5555555555555558LL * ((unsigned __int64)(v24 - v22 - 24) >> 3) + 8];
            v40 = *(_QWORD *)(v15 + 40) - (_QWORD)v21;
            if ( v21 != *(char **)(v15 + 40) )
            {
              v53 = v34;
              v60 = v27;
              v41 = (char *)memcpy(v39, v21, *(_QWORD *)(v15 + 40) - (_QWORD)v21);
              v34 = v53;
              v27 = v60;
              v39 = v41;
            }
            v42 = &v39[v40];
            if ( v27 )
            {
              v61 = v34;
              j_j___libc_free_0(v27, *(_QWORD *)(v15 + 48) - (_QWORD)v27);
              v34 = v61;
            }
            *(_QWORD *)(v15 + 32) = v34;
            *(_QWORD *)(v15 + 40) = v42;
            *(_QWORD *)(v15 + 48) = v67;
            goto LABEL_25;
          }
          v32 = 0x7FFFFFFFFFFFFFF8LL;
          goto LABEL_40;
        }
        do
        {
          if ( v21 )
            *(_QWORD *)v21 = *(_QWORD *)v22;
          v22 += 24;
          v21 += 8;
        }
        while ( v24 != v22 );
        *(_QWORD *)(v15 + 40) += 8 * v25;
LABEL_25:
        v14 = *(_DWORD *)(a2 + 8);
LABEL_26:
        v8 += 16;
        *(_DWORD *)(a2 + 8) = v14 + 1;
      }
      while ( v65 != v8 );
    }
  }
}
