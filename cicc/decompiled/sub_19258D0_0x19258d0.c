// Function: sub_19258D0
// Address: 0x19258d0
//
__int64 __fastcall sub_19258D0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r10
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r9
  int v19; // eax
  int v20; // edx
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // rcx
  int v24; // edi
  __int64 v25; // rcx
  char v26; // si
  char v27; // r9
  bool v28; // si
  __int64 v29; // rcx
  int v30; // edx
  unsigned __int64 v31; // rax
  __int64 v32; // rsi
  unsigned int v33; // esi
  __int64 v34; // rcx
  __int64 v35; // r8
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // rdi
  unsigned __int64 v39; // rax
  char v40; // al
  unsigned int v41; // esi
  __int64 v42; // rcx
  __int64 v43; // r8
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // rdi
  __int64 v47; // r8
  unsigned int v48; // edx
  __int64 v49; // rdi
  int v50; // r10d
  __int64 *v51; // r11
  int v52; // ebx
  int v53; // edx
  int v54; // r10d
  int v55; // r10d
  __int64 *v56; // r11
  int v57; // ebx
  int v58; // r8d
  int v59; // eax
  int v60; // r11d
  __int64 v61; // r10
  unsigned int v62; // edx
  int v63; // edi
  __int64 *v64; // rsi
  int v65; // r10d
  int v66; // r10d
  __int64 *v67; // rsi
  __int64 v68; // r11
  int v69; // edi
  unsigned int v70; // edx
  __int64 v71; // r9
  int v72; // r11d
  __int64 *v73; // r10
  int v74; // edx
  int v75; // edi
  __int64 v76; // [rsp+10h] [rbp-150h]
  __int64 v78; // [rsp+20h] [rbp-140h] BYREF
  __int64 *v79; // [rsp+28h] [rbp-138h] BYREF
  __int64 v80; // [rsp+30h] [rbp-130h] BYREF
  __int64 v81; // [rsp+38h] [rbp-128h]
  unsigned __int64 v82; // [rsp+40h] [rbp-120h]
  __int64 v83; // [rsp+98h] [rbp-C8h]
  __int64 v84; // [rsp+A0h] [rbp-C0h]
  __int64 v85; // [rsp+A8h] [rbp-B8h]
  __int64 v86; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-A8h]
  unsigned __int64 v88; // [rsp+C0h] [rbp-A0h]
  __int64 v89; // [rsp+118h] [rbp-48h]
  __int64 v90; // [rsp+120h] [rbp-40h]
  __int64 v91; // [rsp+128h] [rbp-38h]

  sub_1923B40(&v80, a3);
  sub_1920340(&v86);
  v7 = v83;
  v8 = v84;
  v9 = v89;
  v10 = v90;
  while ( 1 )
  {
    v11 = v9;
    if ( v8 - v7 == v10 - v9 )
      break;
LABEL_3:
    v12 = *(_QWORD *)(v8 - 24);
    if ( v12 == a2 )
    {
      v29 = v8 - 24;
      v8 = v83;
      v84 = v29;
      v7 = v83;
      if ( v29 != v83 )
        goto LABEL_26;
    }
    else
    {
      if ( !*a4 )
        goto LABEL_5;
      v14 = *(unsigned int *)(a1 + 320);
      v78 = v12;
      if ( (_DWORD)v14 )
      {
        v15 = *(_QWORD *)(a1 + 304);
        v16 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v12 == *v17 )
        {
LABEL_16:
          if ( v17 != (__int64 *)(v15 + 16 * v14) )
          {
            if ( *((_BYTE *)v17 + 8) )
              goto LABEL_5;
            goto LABEL_18;
          }
        }
        else
        {
          v30 = 1;
          while ( v18 != -8 )
          {
            v54 = v30 + 1;
            v16 = (v14 - 1) & (v30 + v16);
            v17 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v17;
            if ( v12 == *v17 )
              goto LABEL_16;
            v30 = v54;
          }
        }
      }
      v76 = a1 + 296;
      v31 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v12) + 16) - 34;
      if ( (unsigned int)v31 <= 0x36 && (v32 = 0x40018000000001LL, _bittest64(&v32, v31)) || *(_WORD *)(v78 + 18) )
      {
        v33 = *(_DWORD *)(a1 + 320);
        if ( v33 )
        {
          v34 = v78;
          v35 = *(_QWORD *)(a1 + 304);
          v36 = (v33 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( *v37 == v78 )
            goto LABEL_53;
          v55 = 1;
          v56 = 0;
          while ( v38 != -8 )
          {
            if ( v38 == -16 && !v56 )
              v56 = v37;
            v36 = (v33 - 1) & (v55 + v36);
            v37 = (__int64 *)(v35 + 16LL * v36);
            v38 = *v37;
            if ( v78 == *v37 )
              goto LABEL_53;
            ++v55;
          }
          v57 = *(_DWORD *)(a1 + 312);
          if ( v56 )
            v37 = v56;
          ++*(_QWORD *)(a1 + 296);
          v58 = v57 + 1;
          if ( 4 * (v57 + 1) < 3 * v33 )
          {
            if ( v33 - *(_DWORD *)(a1 + 316) - v58 > v33 >> 3 )
              goto LABEL_80;
            sub_1378900(v76, v33);
            v65 = *(_DWORD *)(a1 + 320);
            if ( v65 )
            {
              v34 = v78;
              v66 = v65 - 1;
              v67 = 0;
              v68 = *(_QWORD *)(a1 + 304);
              v69 = 1;
              v58 = *(_DWORD *)(a1 + 312) + 1;
              v70 = v66 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
              v37 = (__int64 *)(v68 + 16LL * v70);
              v71 = *v37;
              if ( *v37 != v78 )
              {
                while ( v71 != -8 )
                {
                  if ( !v67 && v71 == -16 )
                    v67 = v37;
                  v70 = v66 & (v69 + v70);
                  v37 = (__int64 *)(v68 + 16LL * v70);
                  v71 = *v37;
                  if ( v78 == *v37 )
                    goto LABEL_80;
                  ++v69;
                }
                if ( v67 )
                  v37 = v67;
              }
              goto LABEL_80;
            }
LABEL_136:
            ++*(_DWORD *)(a1 + 312);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 296);
        }
        sub_1378900(v76, 2 * v33);
        v59 = *(_DWORD *)(a1 + 320);
        if ( !v59 )
          goto LABEL_136;
        v60 = v59 - 1;
        v61 = *(_QWORD *)(a1 + 304);
        v58 = *(_DWORD *)(a1 + 312) + 1;
        v62 = (v59 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
        v37 = (__int64 *)(v61 + 16LL * v62);
        v34 = *v37;
        if ( v78 != *v37 )
        {
          v63 = 1;
          v64 = 0;
          while ( v34 != -8 )
          {
            if ( !v64 && v34 == -16 )
              v64 = v37;
            v62 = v60 & (v63 + v62);
            v37 = (__int64 *)(v61 + 16LL * v62);
            v34 = *v37;
            if ( v78 == *v37 )
              goto LABEL_80;
            ++v63;
          }
          v34 = v78;
          if ( v64 )
            v37 = v64;
        }
LABEL_80:
        *(_DWORD *)(a1 + 312) = v58;
        if ( *v37 == -8 )
        {
LABEL_69:
          *v37 = v34;
          *((_BYTE *)v37 + 8) = 0;
LABEL_53:
          *((_BYTE *)v37 + 8) = 1;
          v11 = v89;
LABEL_5:
          if ( v11 )
            j_j___libc_free_0(v11, v91 - v11);
          if ( v88 != v87 )
            _libc_free(v88);
          if ( v83 )
            j_j___libc_free_0(v83, v85 - v83);
          if ( v82 != v81 )
            _libc_free(v82);
          return 1;
        }
LABEL_68:
        --*(_DWORD *)(a1 + 316);
        goto LABEL_69;
      }
      v39 = sub_157EBA0(v78);
      v40 = sub_15F3330(v39);
      v41 = *(_DWORD *)(a1 + 320);
      if ( v40 )
      {
        if ( v41 )
        {
          v34 = v78;
          v47 = *(_QWORD *)(a1 + 304);
          v48 = (v41 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
          v37 = (__int64 *)(v47 + 16LL * v48);
          v49 = *v37;
          if ( *v37 == v78 )
            goto LABEL_53;
          v50 = 1;
          v51 = 0;
          while ( v49 != -8 )
          {
            if ( !v51 && v49 == -16 )
              v51 = v37;
            v48 = (v41 - 1) & (v50 + v48);
            v37 = (__int64 *)(v47 + 16LL * v48);
            v49 = *v37;
            if ( v78 == *v37 )
              goto LABEL_53;
            ++v50;
          }
          v52 = *(_DWORD *)(a1 + 312);
          if ( v51 )
            v37 = v51;
          ++*(_QWORD *)(a1 + 296);
          v53 = v52 + 1;
          if ( 4 * (v52 + 1) < 3 * v41 )
          {
            if ( v41 - *(_DWORD *)(a1 + 316) - v53 > v41 >> 3 )
              goto LABEL_67;
            goto LABEL_110;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 296);
        }
        v41 *= 2;
LABEL_110:
        sub_1378900(v76, v41);
        sub_1923450(v76, &v78, &v79);
        v37 = v79;
        v34 = v78;
        v53 = *(_DWORD *)(a1 + 312) + 1;
LABEL_67:
        *(_DWORD *)(a1 + 312) = v53;
        if ( *v37 == -8 )
          goto LABEL_69;
        goto LABEL_68;
      }
      if ( !v41 )
      {
        ++*(_QWORD *)(a1 + 296);
        goto LABEL_106;
      }
      v42 = v78;
      v43 = *(_QWORD *)(a1 + 304);
      v44 = (v41 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
      v45 = (__int64 *)(v43 + 16LL * v44);
      v46 = *v45;
      if ( v78 != *v45 )
      {
        v72 = 1;
        v73 = 0;
        while ( v46 != -8 )
        {
          if ( v46 == -16 && !v73 )
            v73 = v45;
          v44 = (v41 - 1) & (v72 + v44);
          v45 = (__int64 *)(v43 + 16LL * v44);
          v46 = *v45;
          if ( v78 == *v45 )
            goto LABEL_58;
          ++v72;
        }
        v74 = *(_DWORD *)(a1 + 312);
        if ( v73 )
          v45 = v73;
        ++*(_QWORD *)(a1 + 296);
        v75 = v74 + 1;
        if ( 4 * (v74 + 1) >= 3 * v41 )
        {
LABEL_106:
          v41 *= 2;
        }
        else if ( v41 - *(_DWORD *)(a1 + 316) - v75 > v41 >> 3 )
        {
LABEL_102:
          *(_DWORD *)(a1 + 312) = v75;
          if ( *v45 != -8 )
            --*(_DWORD *)(a1 + 316);
          *v45 = v42;
          *((_BYTE *)v45 + 8) = 0;
          goto LABEL_58;
        }
        sub_1378900(v76, v41);
        sub_1923450(v76, &v78, &v79);
        v45 = v79;
        v42 = v78;
        v75 = *(_DWORD *)(a1 + 312) + 1;
        goto LABEL_102;
      }
LABEL_58:
      *((_BYTE *)v45 + 8) = 0;
LABEL_18:
      if ( v12 != a3 )
      {
        v19 = *(_DWORD *)(a1 + 352);
        if ( v19 )
        {
          v20 = v19 - 1;
          v21 = *(_QWORD *)(a1 + 336);
          v22 = (v19 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v23 = *(_QWORD *)(v21 + 8LL * v22);
          if ( v12 == v23 )
          {
LABEL_21:
            v11 = v89;
            goto LABEL_5;
          }
          v24 = 1;
          while ( v23 != -8 )
          {
            v22 = v20 & (v24 + v22);
            v23 = *(_QWORD *)(v21 + 8LL * v22);
            if ( v12 == v23 )
              goto LABEL_21;
            ++v24;
          }
        }
      }
      if ( *a4 != -1 )
        --*a4;
LABEL_26:
      sub_1923CE0((__int64)&v80);
      v7 = v83;
      v8 = v84;
      v9 = v89;
      v10 = v90;
    }
  }
  if ( v8 != v7 )
  {
    v25 = v9;
    while ( *(_QWORD *)v7 == *(_QWORD *)v25 )
    {
      v26 = *(_BYTE *)(v7 + 16);
      v27 = *(_BYTE *)(v25 + 16);
      if ( v26 && v27 )
        v28 = *(_QWORD *)(v7 + 8) == *(_QWORD *)(v25 + 8);
      else
        v28 = v26 == v27;
      if ( !v28 )
        break;
      v7 += 24;
      v25 += 24;
      if ( v7 == v8 )
        goto LABEL_35;
    }
    goto LABEL_3;
  }
LABEL_35:
  if ( v9 )
    j_j___libc_free_0(v9, v91 - v9);
  if ( v88 != v87 )
    _libc_free(v88);
  if ( v83 )
    j_j___libc_free_0(v83, v85 - v83);
  if ( v82 != v81 )
    _libc_free(v82);
  return 0;
}
