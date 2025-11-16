// Function: sub_1E70F80
// Address: 0x1e70f80
//
unsigned int *__fastcall sub_1E70F80(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  unsigned int *result; // rax
  _QWORD *v6; // rbx
  unsigned int *v7; // r12
  _QWORD *v8; // r14
  unsigned int v9; // r13d
  unsigned int v10; // r9d
  __int64 v11; // r10
  unsigned int v12; // edx
  __int64 v13; // rsi
  _DWORD *v14; // rdi
  __int64 v15; // rcx
  bool v16; // r15
  _QWORD *v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r11
  __int64 v22; // r15
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 *v25; // r8
  int v26; // r9d
  __int64 v27; // r10
  __int64 v28; // rax
  __int64 v29; // rcx
  unsigned __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rcx
  unsigned int v33; // r11d
  __int64 *v34; // rdx
  __int64 v35; // rdi
  unsigned __int64 v36; // r15
  __int64 *v37; // rax
  __int64 v38; // r10
  int v39; // r9d
  __int64 v40; // rcx
  unsigned int v41; // edx
  unsigned int v42; // r10d
  __int64 v43; // rsi
  _DWORD *v44; // rdi
  __int64 v45; // rcx
  _QWORD *v46; // r12
  __int64 *v47; // rbx
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rcx
  unsigned __int64 i; // rax
  __int64 v53; // rdi
  __int64 v54; // rcx
  unsigned int v55; // esi
  __int64 *v56; // rdx
  __int64 v57; // r9
  __int64 *v58; // rdx
  __int64 v59; // rdi
  __int64 v60; // rsi
  int v61; // edx
  unsigned int v62; // eax
  __int64 v63; // rsi
  int v64; // edx
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned __int64 v67; // r15
  int v68; // eax
  __int64 v69; // r15
  __int64 *v70; // rax
  __int64 v71; // rdx
  _QWORD *v72; // rdi
  _QWORD *v73; // rdx
  __int64 v74; // rcx
  int v75; // r8d
  int v76; // [rsp+Ch] [rbp-54h]
  unsigned int v77; // [rsp+Ch] [rbp-54h]
  int v78; // [rsp+Ch] [rbp-54h]
  unsigned int v79; // [rsp+Ch] [rbp-54h]
  __int64 v80; // [rsp+10h] [rbp-50h]
  unsigned int *v81; // [rsp+10h] [rbp-50h]
  __int64 v82; // [rsp+10h] [rbp-50h]
  __int64 v83; // [rsp+18h] [rbp-48h]
  __int64 v84; // [rsp+18h] [rbp-48h]
  __int64 v85; // [rsp+18h] [rbp-48h]
  unsigned int *v86; // [rsp+20h] [rbp-40h]
  __int64 *v87; // [rsp+20h] [rbp-40h]
  unsigned __int64 v88; // [rsp+20h] [rbp-40h]
  _QWORD *v89; // [rsp+20h] [rbp-40h]
  __int64 *v90; // [rsp+20h] [rbp-40h]
  int v91; // [rsp+20h] [rbp-40h]
  __int64 v92; // [rsp+20h] [rbp-40h]
  unsigned int *v93; // [rsp+28h] [rbp-38h]

  result = &a2[2 * a3];
  v93 = result;
  if ( result != a2 )
  {
    v6 = a1;
    v7 = a2;
    v8 = a1 + 43;
    while ( 1 )
    {
      v9 = *v7;
      if ( (*v7 & 0x80000000) == 0 )
        goto LABEL_18;
      v10 = v9 & 0x7FFFFFFF;
      v11 = v9 & 0x7FFFFFFF;
      if ( !*((_BYTE *)v6 + 2569) )
        break;
      LODWORD(a5) = *((_DWORD *)v6 + 582);
      result = (unsigned int *)v6[316];
      v12 = *((unsigned __int8 *)result + v11);
      if ( v12 < (unsigned int)a5 )
      {
        v13 = v6[290];
        while ( 1 )
        {
          result = (unsigned int *)v12;
          v14 = (_DWORD *)(v13 + 24LL * v12);
          if ( (*v14 & 0x7FFFFFFF) == v10 )
          {
            v15 = (unsigned int)v14[4];
            if ( (_DWORD)v15 != -1 && *(_DWORD *)(v13 + 24 * v15 + 20) == -1 )
              break;
          }
          v12 += 256;
          if ( (unsigned int)a5 <= v12 )
            goto LABEL_18;
        }
        if ( v12 != -1 )
        {
          v86 = v7;
          v16 = v7[1] != 0;
          v17 = v6;
          do
          {
            v18 = 24LL * (_QWORD)result;
            v19 = v13 + 24LL * (_QWORD)result;
            v20 = *(_QWORD *)(v19 + 8);
            if ( (*(_BYTE *)(v20 + 229) & 4) == 0 && (_QWORD *)v20 != v8 )
            {
              sub_1EE7080(v17[319] + ((unsigned __int64)*(unsigned int *)(v20 + 192) << 6), v9, v16, v17[5]);
              v13 = v17[290];
              v19 = v13 + v18;
            }
            result = (unsigned int *)*(unsigned int *)(v19 + 20);
          }
          while ( (_DWORD)result != -1 );
          v6 = v17;
          v7 = v86;
        }
      }
LABEL_18:
      v7 += 2;
      if ( v93 == v7 )
        return result;
    }
    v21 = v6[264];
    v22 = 8 * v11;
    v23 = *(unsigned int *)(v21 + 408);
    if ( (unsigned int)v23 > v10 )
    {
      a5 = *(__int64 **)(*(_QWORD *)(v21 + 400) + 8 * v11);
      if ( a5 )
      {
LABEL_22:
        v24 = sub_1E6BEE0(v6[480], v6[115] + 24LL);
        v28 = v6[115];
        if ( v24 == v28 + 24 )
        {
          v66 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v6[264] + 272LL) + 392LL) + 16LL * *(unsigned int *)(v28 + 48) + 8);
          v67 = v66 & 0xFFFFFFFFFFFFFFF8LL;
          v68 = (v66 >> 1) & 3;
          if ( v68 )
            v69 = (2LL * (v68 - 1)) | v67;
          else
            v69 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL | 6;
          v78 = v26;
          v82 = v27;
          v90 = v25;
          v70 = (__int64 *)sub_1DB3C70(v25, v69);
          a5 = v90;
          v38 = v82;
          v83 = 0;
          v39 = v78;
          if ( v70 != (__int64 *)(*v90 + 24LL * *((unsigned int *)v90 + 2))
            && (*(_DWORD *)((*v70 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v70 >> 1) & 3)) <= (*(_DWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v69 >> 1) & 3) )
          {
            v83 = v70[2];
          }
LABEL_29:
          result = (unsigned int *)v6[316];
          v41 = *((unsigned __int8 *)result + v38);
          v42 = *((_DWORD *)v6 + 582);
          if ( v41 >= v42 )
            goto LABEL_18;
          v43 = v6[290];
          while ( 1 )
          {
            result = (unsigned int *)v41;
            v44 = (_DWORD *)(v43 + 24LL * v41);
            if ( (*v44 & 0x7FFFFFFF) == v39 )
            {
              v45 = (unsigned int)v44[4];
              if ( (_DWORD)v45 != -1 && *(_DWORD *)(v43 + 24 * v45 + 20) == -1 )
                break;
            }
            v41 += 256;
            if ( v42 <= v41 )
              goto LABEL_18;
          }
          if ( v41 == -1 )
            goto LABEL_18;
          v77 = v9;
          v81 = v7;
          v46 = v6;
          v47 = a5;
          while ( 1 )
          {
            v48 = 24LL * (_QWORD)result;
            v49 = v43 + 24LL * (_QWORD)result;
            v50 = *(_QWORD *)(v49 + 8);
            if ( (*(_BYTE *)(v50 + 229) & 4) == 0 && (_QWORD *)v50 != v8 )
              break;
LABEL_51:
            result = (unsigned int *)*(unsigned int *)(v49 + 20);
            if ( (_DWORD)result == -1 )
            {
              v6 = v46;
              v7 = v81;
              goto LABEL_18;
            }
          }
          v51 = *(_QWORD *)(v46[264] + 272LL);
          for ( i = *(_QWORD *)(v50 + 8); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
            ;
          v53 = *(_QWORD *)(v51 + 368);
          v54 = *(unsigned int *)(v51 + 384);
          if ( (_DWORD)v54 )
          {
            v55 = (v54 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
            v56 = (__int64 *)(v53 + 16LL * v55);
            v57 = *v56;
            if ( *v56 == i )
            {
LABEL_43:
              v88 = v56[1] & 0xFFFFFFFFFFFFFFF8LL;
              v58 = (__int64 *)sub_1DB3C70(v47, v88);
              v59 = *v47 + 24LL * *((unsigned int *)v47 + 2);
              v60 = 0;
              if ( v58 != (__int64 *)v59
                && (*(_DWORD *)((*v58 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v58 >> 1) & 3) <= *(_DWORD *)(v88 + 24) )
              {
                v60 = v58[2];
                if ( (v88 != (v58[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v59 != v58 + 3)
                  && v88 == *(_QWORD *)(v60 + 8) )
                {
                  v60 = 0;
                }
              }
              if ( v60 == v83 )
                sub_1EE7080(v46[319] + ((unsigned __int64)*(unsigned int *)(v50 + 192) << 6), v77, 1, v46[5]);
              v43 = v46[290];
              v49 = v43 + v48;
              goto LABEL_51;
            }
            v61 = 1;
            while ( v57 != -8 )
            {
              v75 = v61 + 1;
              v55 = (v54 - 1) & (v61 + v55);
              v56 = (__int64 *)(v53 + 16LL * v55);
              v57 = *v56;
              if ( *v56 == i )
                goto LABEL_43;
              v61 = v75;
            }
          }
          v56 = (__int64 *)(v53 + 16 * v54);
          goto LABEL_43;
        }
        v29 = *(_QWORD *)(v6[264] + 272LL);
        v30 = v24;
        if ( (*(_BYTE *)(v24 + 46) & 4) != 0 )
        {
          do
            v30 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v30 + 46) & 4) != 0 );
        }
        v31 = *(_QWORD *)(v29 + 368);
        v32 = *(unsigned int *)(v29 + 384);
        if ( (_DWORD)v32 )
        {
          v33 = (v32 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v34 = (__int64 *)(v31 + 16LL * v33);
          v35 = *v34;
          if ( *v34 == v30 )
          {
LABEL_27:
            v76 = v26;
            v80 = v27;
            v36 = v34[1] & 0xFFFFFFFFFFFFFFF8LL;
            v87 = v25;
            v37 = (__int64 *)sub_1DB3C70(v25, v36);
            a5 = v87;
            v38 = v80;
            v83 = 0;
            v39 = v76;
            v40 = *v87 + 24LL * *((unsigned int *)v87 + 2);
            if ( v37 != (__int64 *)v40
              && (*(_DWORD *)((*v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v37 >> 1) & 3) <= *(_DWORD *)(v36 + 24) )
            {
              v83 = v37[2];
              if ( v36 != (v37[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v40 != v37 + 3 )
              {
                v65 = 0;
                if ( v36 != *(_QWORD *)(v83 + 8) )
                  v65 = v83;
                v83 = v65;
              }
            }
            goto LABEL_29;
          }
          v64 = 1;
          while ( v35 != -8 )
          {
            v33 = (v32 - 1) & (v64 + v33);
            v91 = v64 + 1;
            v34 = (__int64 *)(v31 + 16LL * v33);
            v35 = *v34;
            if ( *v34 == v30 )
              goto LABEL_27;
            v64 = v91;
          }
        }
        v34 = (__int64 *)(v31 + 16 * v32);
        goto LABEL_27;
      }
    }
    v62 = v10 + 1;
    if ( (unsigned int)v23 < v10 + 1 )
    {
      v71 = v62;
      if ( v62 < v23 )
      {
        *(_DWORD *)(v21 + 408) = v62;
      }
      else if ( v62 > v23 )
      {
        if ( v62 > (unsigned __int64)*(unsigned int *)(v21 + 412) )
        {
          v79 = v10 + 1;
          v85 = v6[264];
          v92 = v62;
          sub_16CD150(v21 + 400, (const void *)(v21 + 416), v62, 8, (int)a5, v10);
          v21 = v85;
          v62 = v79;
          v11 = v9 & 0x7FFFFFFF;
          v23 = *(unsigned int *)(v85 + 408);
          v71 = v92;
        }
        v63 = *(_QWORD *)(v21 + 400);
        v72 = (_QWORD *)(v63 + 8 * v71);
        v73 = (_QWORD *)(v63 + 8 * v23);
        v74 = *(_QWORD *)(v21 + 416);
        if ( v72 != v73 )
        {
          do
            *v73++ = v74;
          while ( v72 != v73 );
          v63 = *(_QWORD *)(v21 + 400);
        }
        *(_DWORD *)(v21 + 408) = v62;
        goto LABEL_58;
      }
    }
    v63 = *(_QWORD *)(v21 + 400);
LABEL_58:
    v84 = v11;
    v89 = (_QWORD *)v21;
    *(_QWORD *)(v63 + v22) = sub_1DBA290(v9);
    sub_1DBB110(v89, *(_QWORD *)(v89[50] + 8 * v84));
    goto LABEL_22;
  }
  return result;
}
