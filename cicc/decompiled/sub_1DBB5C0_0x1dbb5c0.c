// Function: sub_1DBB5C0
// Address: 0x1dbb5c0
//
void __fastcall sub_1DBB5C0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  unsigned __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // r8
  __int64 v13; // r10
  __int64 v14; // rbx
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // r12
  __int64 v24; // r13
  unsigned __int64 v25; // rdi
  unsigned int v26; // ebx
  __int64 v27; // rcx
  __int64 *v28; // rax
  char v29; // dl
  __int64 *v30; // rsi
  __int64 *v31; // rcx
  __int64 *v32; // r12
  char v33; // dl
  __int64 v34; // r8
  unsigned __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 *v37; // rcx
  int v38; // r9d
  __int64 v39; // r8
  __int64 v40; // r10
  __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // r13
  __int64 *v44; // rax
  __int64 *v45; // rdi
  __int64 *v46; // rcx
  __int64 *v47; // r12
  __int64 v48; // rsi
  char v49; // dl
  __int64 v50; // r8
  unsigned __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 *v53; // rcx
  int v54; // r9d
  __int64 v55; // r8
  __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 *v58; // rax
  __int64 *v59; // rdi
  __int64 *v60; // rcx
  __int64 v61; // rax
  _QWORD *v62; // rsi
  _QWORD *v63; // rax
  __int64 v64; // rdx
  __int128 v65; // [rsp-20h] [rbp-1C0h]
  const void *v66; // [rsp+8h] [rbp-198h]
  __int64 v67; // [rsp+10h] [rbp-190h]
  __int64 v68; // [rsp+10h] [rbp-190h]
  __int64 v69; // [rsp+10h] [rbp-190h]
  __int64 v70; // [rsp+10h] [rbp-190h]
  __int64 v71; // [rsp+18h] [rbp-188h]
  __int64 v72; // [rsp+18h] [rbp-188h]
  __int64 *v73; // [rsp+18h] [rbp-188h]
  __int64 v74; // [rsp+18h] [rbp-188h]
  __int64 v76; // [rsp+28h] [rbp-178h]
  __int64 v77; // [rsp+28h] [rbp-178h]
  __int64 *v78; // [rsp+28h] [rbp-178h]
  __int64 v79; // [rsp+28h] [rbp-178h]
  __int64 v80; // [rsp+28h] [rbp-178h]
  __int64 v81; // [rsp+50h] [rbp-150h] BYREF
  __int64 *v82; // [rsp+58h] [rbp-148h]
  __int64 *v83; // [rsp+60h] [rbp-140h]
  __int64 v84; // [rsp+68h] [rbp-138h]
  int v85; // [rsp+70h] [rbp-130h]
  _BYTE v86[72]; // [rsp+78h] [rbp-128h] BYREF
  __int64 v87; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 *v88; // [rsp+C8h] [rbp-D8h]
  __int64 *v89; // [rsp+D0h] [rbp-D0h]
  __int64 v90; // [rsp+D8h] [rbp-C8h]
  int v91; // [rsp+E0h] [rbp-C0h]
  _BYTE v92[184]; // [rsp+E8h] [rbp-B8h] BYREF

  v82 = (__int64 *)v86;
  v10 = *(unsigned int *)(a1 + 408);
  v83 = (__int64 *)v86;
  v88 = (__int64 *)v92;
  v89 = (__int64 *)v92;
  v11 = a4 & 0x7FFFFFFF;
  v81 = 0;
  v12 = a4 & 0x7FFFFFFF;
  v84 = 8;
  v13 = 8 * v12;
  v85 = 0;
  v87 = 0;
  v90 = 16;
  v91 = 0;
  if ( (a4 & 0x7FFFFFFFu) >= (unsigned int)v10 || (v14 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8LL * v11)) == 0 )
  {
    v26 = v11 + 1;
    if ( (unsigned int)v10 < v11 + 1 )
    {
      v61 = v26;
      if ( v26 < v10 )
      {
        *(_DWORD *)(a1 + 408) = v26;
      }
      else if ( v26 > v10 )
      {
        if ( v26 > (unsigned __int64)*(unsigned int *)(a1 + 412) )
        {
          v70 = a4 & 0x7FFFFFFF;
          sub_16CD150(a1 + 400, (const void *)(a1 + 416), v26, 8, v12, a6);
          v10 = *(unsigned int *)(a1 + 408);
          v12 = v70;
          v13 = 8 * v70;
          v61 = v26;
        }
        v27 = *(_QWORD *)(a1 + 400);
        v62 = (_QWORD *)(v27 + 8 * v61);
        v63 = (_QWORD *)(v27 + 8 * v10);
        v64 = *(_QWORD *)(a1 + 416);
        if ( v62 != v63 )
        {
          do
            *v63++ = v64;
          while ( v62 != v63 );
          v27 = *(_QWORD *)(a1 + 400);
        }
        *(_DWORD *)(a1 + 408) = v26;
        goto LABEL_16;
      }
    }
    v27 = *(_QWORD *)(a1 + 400);
LABEL_16:
    v77 = v12;
    *(_QWORD *)(v27 + v13) = sub_1DBA290(a4);
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v77);
    sub_1DBB110((_QWORD *)a1, v14);
  }
  if ( a5 )
  {
    do
      v14 = *(_QWORD *)(v14 + 104);
    while ( (*(_DWORD *)(v14 + 112) & a5) == 0 );
  }
  v15 = *(_DWORD *)(a3 + 8);
  if ( v15 )
  {
    v66 = (const void *)(a3 + 16);
    do
    {
      while ( 1 )
      {
        v22 = (__int64 *)(*(_QWORD *)a3 + 16LL * v15 - 16);
        v23 = *v22;
        v24 = v22[1];
        *(_DWORD *)(a3 + 8) = v15 - 1;
        v25 = v23 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((v23 >> 1) & 3) != 0 )
          v16 = v25 | (2LL * (int)(((v23 >> 1) & 3) - 1));
        else
          v16 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL | 6;
        v76 = *(_QWORD *)(a1 + 272);
        v71 = sub_1DA9310(v76, v16);
        v17 = *(_QWORD *)(*(_QWORD *)(v76 + 392) + 16LL * *(unsigned int *)(v71 + 48));
        if ( !sub_1DB7FE0(a2, v17, v23) )
        {
          *((_QWORD *)&v65 + 1) = v23;
          *(_QWORD *)&v65 = v17;
          sub_1DB8610(a2, v17, v18, v17, v19, v20, v65, v24);
          v47 = *(__int64 **)(v71 + 64);
          v73 = *(__int64 **)(v71 + 72);
          if ( v73 == v47 )
            goto LABEL_11;
          while ( 2 )
          {
            v48 = *v47;
            v58 = v88;
            if ( v89 != v88 )
              goto LABEL_56;
            v59 = &v88[HIDWORD(v90)];
            if ( v88 != v59 )
            {
              v60 = 0;
              while ( v48 != *v58 )
              {
                if ( *v58 == -2 )
                  v60 = v58;
                if ( v59 == ++v58 )
                {
                  if ( !v60 )
                    goto LABEL_85;
                  *v60 = v48;
                  --v91;
                  ++v87;
                  goto LABEL_57;
                }
              }
LABEL_65:
              if ( v73 == ++v47 )
                goto LABEL_11;
              continue;
            }
            break;
          }
LABEL_85:
          if ( HIDWORD(v90) < (unsigned int)v90 )
          {
            ++HIDWORD(v90);
            *v59 = v48;
            ++v87;
          }
          else
          {
LABEL_56:
            v79 = *v47;
            sub_16CCBA0((__int64)&v87, v48);
            v48 = v79;
            if ( !v49 )
              goto LABEL_65;
          }
LABEL_57:
          v50 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL) + 16LL * *(unsigned int *)(v48 + 48) + 8);
          v51 = v50 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v50 >> 1) & 3) != 0 )
            v52 = (2LL * (int)(((v50 >> 1) & 3) - 1)) | v51;
          else
            v52 = *(_QWORD *)v51 & 0xFFFFFFFFFFFFFFF8LL | 6;
          v68 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL) + 16LL * *(unsigned int *)(v48 + 48) + 8);
          v80 = v52;
          v53 = (__int64 *)sub_1DB3C70((__int64 *)v14, v52);
          if ( v53 != (__int64 *)(*(_QWORD *)v14 + 24LL * *(unsigned int *)(v14 + 8)) )
          {
            v55 = v68;
            if ( (*(_DWORD *)((*v53 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v53 >> 1) & 3) <= (*(_DWORD *)((v80 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v80 >> 1) & 3) )
            {
              if ( v53[2] )
              {
                v56 = *(unsigned int *)(a3 + 8);
                if ( (unsigned int)v56 >= *(_DWORD *)(a3 + 12) )
                {
                  sub_16CD150(a3, v66, 0, 16, v68, v54);
                  v56 = *(unsigned int *)(a3 + 8);
                  v55 = v68;
                }
                v57 = (_QWORD *)(*(_QWORD *)a3 + 16 * v56);
                *v57 = v55;
                v57[1] = v24;
                ++*(_DWORD *)(a3 + 8);
              }
            }
          }
          goto LABEL_65;
        }
        v21 = *(_QWORD *)(v24 + 8);
        if ( (v21 & 6) != 0 || v17 != v21 )
          goto LABEL_11;
        v28 = v82;
        if ( v83 == v82 )
          break;
LABEL_18:
        sub_16CCBA0((__int64)&v81, v24);
        if ( v29 )
          goto LABEL_33;
        v15 = *(_DWORD *)(a3 + 8);
        if ( !v15 )
          goto LABEL_20;
      }
      v30 = &v82[HIDWORD(v84)];
      if ( v82 == v30 )
        goto LABEL_90;
      v31 = 0;
      do
      {
        if ( v24 == *v28 )
          goto LABEL_11;
        if ( *v28 == -2 )
          v31 = v28;
        ++v28;
      }
      while ( v30 != v28 );
      if ( !v31 )
      {
LABEL_90:
        if ( HIDWORD(v84) >= (unsigned int)v84 )
          goto LABEL_18;
        ++HIDWORD(v84);
        *v30 = v24;
        ++v81;
      }
      else
      {
        *v31 = v24;
        --v85;
        ++v81;
      }
LABEL_33:
      v32 = *(__int64 **)(v71 + 64);
      v78 = *(__int64 **)(v71 + 72);
      if ( v78 != v32 )
      {
        while ( 1 )
        {
          v43 = *v32;
          v44 = v88;
          if ( v89 == v88 )
          {
            v45 = &v88[HIDWORD(v90)];
            if ( v88 != v45 )
            {
              v46 = 0;
              while ( v43 != *v44 )
              {
                if ( *v44 == -2 )
                  v46 = v44;
                if ( v45 == ++v44 )
                {
                  if ( !v46 )
                    goto LABEL_87;
                  *v46 = v43;
                  --v91;
                  ++v87;
                  goto LABEL_36;
                }
              }
              goto LABEL_44;
            }
LABEL_87:
            if ( HIDWORD(v90) < (unsigned int)v90 )
              break;
          }
          sub_16CCBA0((__int64)&v87, *v32);
          if ( v33 )
            goto LABEL_36;
LABEL_44:
          if ( v78 == ++v32 )
            goto LABEL_11;
        }
        ++HIDWORD(v90);
        *v45 = v43;
        ++v87;
LABEL_36:
        v34 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL) + 16LL * *(unsigned int *)(v43 + 48) + 8);
        v35 = v34 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((v34 >> 1) & 3) != 0 )
          v36 = (2LL * (int)(((v34 >> 1) & 3) - 1)) | v35;
        else
          v36 = *(_QWORD *)v35 & 0xFFFFFFFFFFFFFFF8LL | 6;
        v67 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL) + 16LL * *(unsigned int *)(v43 + 48) + 8);
        v72 = v36;
        v37 = (__int64 *)sub_1DB3C70((__int64 *)v14, v36);
        if ( v37 != (__int64 *)(*(_QWORD *)v14 + 24LL * *(unsigned int *)(v14 + 8)) )
        {
          v39 = v67;
          if ( (*(_DWORD *)((*v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v37 >> 1) & 3) <= (*(_DWORD *)((v72 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v72 >> 1) & 3) )
          {
            v40 = v37[2];
            if ( v40 )
            {
              v41 = *(unsigned int *)(a3 + 8);
              if ( (unsigned int)v41 >= *(_DWORD *)(a3 + 12) )
              {
                v69 = v37[2];
                v74 = v39;
                sub_16CD150(a3, v66, 0, 16, v39, v38);
                v41 = *(unsigned int *)(a3 + 8);
                v40 = v69;
                v39 = v74;
              }
              v42 = (_QWORD *)(*(_QWORD *)a3 + 16 * v41);
              *v42 = v39;
              v42[1] = v40;
              ++*(_DWORD *)(a3 + 8);
            }
          }
        }
        goto LABEL_44;
      }
LABEL_11:
      v15 = *(_DWORD *)(a3 + 8);
    }
    while ( v15 );
  }
LABEL_20:
  if ( v89 != v88 )
    _libc_free((unsigned __int64)v89);
  if ( v83 != v82 )
    _libc_free((unsigned __int64)v83);
}
