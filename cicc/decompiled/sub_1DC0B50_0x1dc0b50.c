// Function: sub_1DC0B50
// Address: 0x1dc0b50
//
void __fastcall sub_1DC0B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r13
  __int64 *v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // esi
  unsigned int v11; // ecx
  __int64 v12; // r13
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 *v18; // rsi
  __int64 *v19; // rdx
  __int64 *v20; // rax
  __int64 *v21; // rdi
  __int64 v22; // r13
  unsigned __int64 *v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rsi
  unsigned __int64 v26; // r8
  unsigned __int64 v27; // r13
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  char v32; // cl
  unsigned __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // r15
  __int64 v36; // r14
  unsigned __int64 v37; // r13
  __int64 *v38; // rdx
  __int64 v39; // rdi
  unsigned int v40; // r8d
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 v43; // r9
  int v44; // r8d
  int v45; // r9d
  __int64 v46; // rax
  unsigned __int64 v47; // r15
  __int64 v48; // r13
  __int64 *v49; // rax
  char v50; // dl
  __int64 v51; // rdi
  __int64 v52; // r14
  __int64 *v53; // rax
  __int64 *v54; // rcx
  unsigned int v55; // r9d
  __int64 *v56; // rsi
  int v57; // r8d
  int v58; // r9d
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 *v61; // rsi
  __int64 *v62; // rcx
  int v63; // r8d
  int v64; // r9d
  __int64 v65; // rax
  __int64 *v66; // [rsp+8h] [rbp-158h]
  __int64 v67; // [rsp+10h] [rbp-150h]
  __int64 v68; // [rsp+20h] [rbp-140h]
  __int64 *v69; // [rsp+20h] [rbp-140h]
  __int64 v70; // [rsp+20h] [rbp-140h]
  __int64 v71; // [rsp+28h] [rbp-138h]
  __int64 v72; // [rsp+28h] [rbp-138h]
  __int64 v73; // [rsp+28h] [rbp-138h]
  __int64 v74; // [rsp+30h] [rbp-130h]
  __int64 v76; // [rsp+40h] [rbp-120h] BYREF
  unsigned __int64 v77; // [rsp+48h] [rbp-118h] BYREF
  unsigned __int64 v78; // [rsp+50h] [rbp-110h]
  unsigned __int64 v79; // [rsp+58h] [rbp-108h]
  __int64 *v80; // [rsp+60h] [rbp-100h] BYREF
  unsigned __int64 v81; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v82; // [rsp+70h] [rbp-F0h]
  __int64 v83; // [rsp+78h] [rbp-E8h]
  __int64 v84; // [rsp+80h] [rbp-E0h] BYREF
  __int64 *v85; // [rsp+88h] [rbp-D8h]
  __int64 *v86; // [rsp+90h] [rbp-D0h]
  __int64 v87; // [rsp+98h] [rbp-C8h]
  int v88; // [rsp+A0h] [rbp-C0h]
  _BYTE v89[184]; // [rsp+A8h] [rbp-B8h] BYREF

  v5 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v8 = (__int64 *)sub_1DB3C70((__int64 *)a2, a3 & 0xFFFFFFFFFFFFFFF8LL);
  v9 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( v8 != (__int64 *)v9 )
  {
    v10 = *(_DWORD *)(v5 + 24);
    v11 = *(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)(v11 | (*v8 >> 1) & 3) <= v10 && v5 == (v8[1] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( (__int64 *)v9 == v8 + 3 )
        return;
      v11 = *(_DWORD *)((v8[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v8 += 3;
    }
    if ( v10 >= v11 )
    {
      v12 = v8[1];
      v74 = v8[2];
      if ( v74 )
      {
        v71 = *(_QWORD *)(a1 + 272);
        v13 = sub_1DA9310(v71, a3);
        v14 = *(_QWORD *)(*(_QWORD *)(v71 + 392) + 16LL * *(unsigned int *)(v13 + 48) + 8);
        if ( (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12 >> 1) & 3) < (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(v14 >> 1)
                                                                                              & 3) )
        {
          sub_1DB4410(a2, a3, v12, 0);
          if ( a4 )
          {
            v65 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v65 >= *(_DWORD *)(a4 + 12) )
            {
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v63, v64);
              v65 = *(unsigned int *)(a4 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v65) = v12;
            ++*(_DWORD *)(a4 + 8);
          }
        }
        else
        {
          v68 = v13;
          v72 = *(_QWORD *)(*(_QWORD *)(v71 + 392) + 16LL * *(unsigned int *)(v13 + 48) + 8);
          sub_1DB4410(a2, a3, v14, 0);
          v15 = v68;
          if ( a4 )
          {
            v16 = *(unsigned int *)(a4 + 8);
            v17 = v72;
            if ( (unsigned int)v16 >= *(_DWORD *)(a4 + 12) )
            {
              v70 = v72;
              v73 = v15;
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v15, v17);
              v16 = *(unsigned int *)(a4 + 8);
              v17 = v70;
              v15 = v73;
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v16) = v17;
            ++*(_DWORD *)(a4 + 8);
          }
          v18 = *(__int64 **)(v15 + 96);
          v19 = *(__int64 **)(v15 + 88);
          v20 = (__int64 *)v89;
          v84 = 0;
          v85 = (__int64 *)v89;
          v86 = (__int64 *)v89;
          v87 = 16;
          v88 = 0;
          v66 = v18;
          if ( v19 != v18 )
          {
            v69 = v19;
            v21 = (__int64 *)v89;
            while ( 1 )
            {
              v22 = *v69;
              v81 = 0;
              v82 = 0;
              v80 = &v84;
              v83 = 0;
              if ( v21 == v20 )
              {
                v61 = &v20[HIDWORD(v87)];
                v23 = (unsigned __int64 *)HIDWORD(v87);
                if ( v61 != v20 )
                {
                  v62 = 0;
                  do
                  {
                    v24 = *v20;
                    if ( v22 == *v20 )
                      goto LABEL_15;
                    if ( v24 == -2 )
                      v62 = v20;
                    ++v20;
                  }
                  while ( v61 != v20 );
                  if ( v62 )
                  {
                    *v62 = v22;
                    --v88;
                    ++v84;
LABEL_87:
                    v23 = &v81;
                    v76 = v22;
                    LOBYTE(v78) = 0;
                    sub_1BFDD10(&v81, (__int64)&v76);
                    goto LABEL_15;
                  }
                }
                if ( HIDWORD(v87) < (unsigned int)v87 )
                {
                  ++HIDWORD(v87);
                  *v61 = v22;
                  ++v84;
                  goto LABEL_87;
                }
              }
              v23 = (unsigned __int64 *)&v84;
              sub_16CCBA0((__int64)&v84, v22);
              if ( (_BYTE)v24 )
                goto LABEL_87;
LABEL_15:
              v25 = v82;
              v77 = 0;
              v26 = v81;
              v78 = 0;
              v76 = (__int64)v80;
              v79 = 0;
              v27 = v82 - v81;
              if ( v82 == v81 )
              {
                v29 = 0;
              }
              else
              {
                if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
                  sub_4261EA(v23, v82, v24);
                v28 = sub_22077B0(v82 - v81);
                v25 = v82;
                v26 = v81;
                v29 = v28;
              }
              v77 = v29;
              v78 = v29;
              v79 = v29 + v27;
              if ( v26 == v25 )
              {
                v33 = v29;
              }
              else
              {
                v30 = v29;
                v31 = v26;
                do
                {
                  if ( v30 )
                  {
                    *(_QWORD *)v30 = *(_QWORD *)v31;
                    v32 = *(_BYTE *)(v31 + 16);
                    *(_BYTE *)(v30 + 16) = v32;
                    if ( v32 )
                      *(_QWORD *)(v30 + 8) = *(_QWORD *)(v31 + 8);
                  }
                  v31 += 24LL;
                  v30 += 24LL;
                }
                while ( v31 != v25 );
                v33 = v29 + 8 * ((v31 - 24 - v26) >> 3) + 24;
              }
              v78 = v33;
              if ( v26 )
              {
                j_j___libc_free_0(v26, v83 - v26);
                v29 = v77;
                v33 = v78;
              }
              while ( v29 != v33 )
              {
LABEL_28:
                v34 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL)
                                + 16LL * *(unsigned int *)(*(_QWORD *)(v33 - 24) + 48LL));
                v35 = *v34;
                v36 = v34[1];
                v37 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
                v38 = (__int64 *)sub_1DB3C70((__int64 *)a2, v37);
                v39 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
                if ( v38 == (__int64 *)v39 )
                  goto LABEL_68;
                v40 = *(_DWORD *)(v37 + 24);
                v41 = *(_DWORD *)((*v38 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                if ( (unsigned __int64)(v41 | (*v38 >> 1) & 3) > v40 )
                {
                  v42 = 0;
                  v43 = 0;
                  goto LABEL_33;
                }
                v42 = v38[1];
                v43 = v38[2];
                if ( v37 != (v42 & 0xFFFFFFFFFFFFFFF8LL) )
                  goto LABEL_31;
                if ( (__int64 *)v39 != v38 + 3 )
                {
                  v41 = *(_DWORD *)((v38[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
                  v38 += 3;
LABEL_31:
                  if ( v37 == *(_QWORD *)(v43 + 8) )
                    v43 = 0;
LABEL_33:
                  if ( v41 <= v40 )
                    v42 = v38[1];
                }
                if ( v74 != v43 )
                {
LABEL_68:
                  v78 -= 24LL;
                  v29 = v77;
                  if ( v78 == v77 )
                    goto LABEL_67;
                  goto LABEL_69;
                }
                if ( (*(_DWORD *)((v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v42 >> 1) & 3) < (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v36 >> 1) & 3) )
                {
                  v67 = v42;
                  sub_1DB4410(a2, v35, v42, 0);
                  if ( a4 )
                  {
                    v59 = *(unsigned int *)(a4 + 8);
                    v60 = v67;
                    if ( (unsigned int)v59 >= *(_DWORD *)(a4 + 12) )
                    {
                      sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v57, v58);
                      v59 = *(unsigned int *)(a4 + 8);
                      v60 = v67;
                    }
                    *(_QWORD *)(*(_QWORD *)a4 + 8 * v59) = v60;
                    ++*(_DWORD *)(a4 + 8);
                  }
                  v29 = v77;
                  v78 -= 24LL;
                  v33 = v77;
                  if ( v78 == v77 )
                    continue;
LABEL_69:
                  sub_1DC0A30((unsigned __int64 *)&v76);
                  v29 = v77;
                  v33 = v78;
                  continue;
                }
                sub_1DB4410(a2, v35, v36, 0);
                if ( a4 )
                {
                  v46 = *(unsigned int *)(a4 + 8);
                  if ( (unsigned int)v46 >= *(_DWORD *)(a4 + 12) )
                  {
                    sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v44, v45);
                    v46 = *(unsigned int *)(a4 + 8);
                  }
                  *(_QWORD *)(*(_QWORD *)a4 + 8 * v46) = v36;
                  ++*(_DWORD *)(a4 + 8);
                }
                v47 = v78;
                while ( 2 )
                {
                  v48 = *(_QWORD *)(v47 - 24);
                  if ( !*(_BYTE *)(v47 - 8) )
                  {
                    v49 = *(__int64 **)(v48 + 88);
                    *(_BYTE *)(v47 - 8) = 1;
                    *(_QWORD *)(v47 - 16) = v49;
                    goto LABEL_46;
                  }
                  while ( 1 )
                  {
                    v49 = *(__int64 **)(v47 - 16);
LABEL_46:
                    if ( v49 == *(__int64 **)(v48 + 96) )
                      break;
                    *(_QWORD *)(v47 - 16) = v49 + 1;
                    v51 = v76;
                    v52 = *v49;
                    v53 = *(__int64 **)(v76 + 8);
                    if ( *(__int64 **)(v76 + 16) != v53 )
                      goto LABEL_44;
                    v54 = &v53[*(unsigned int *)(v76 + 28)];
                    v55 = *(_DWORD *)(v76 + 28);
                    if ( v53 == v54 )
                    {
LABEL_64:
                      if ( v55 < *(_DWORD *)(v76 + 24) )
                      {
                        *(_DWORD *)(v76 + 28) = v55 + 1;
                        *v54 = v52;
                        ++*(_QWORD *)v51;
LABEL_55:
                        v80 = (__int64 *)v52;
                        LOBYTE(v82) = 0;
                        sub_1BFDD10(&v77, (__int64)&v80);
                        v29 = v77;
                        v33 = v78;
                        if ( v77 != v78 )
                          goto LABEL_28;
                        goto LABEL_56;
                      }
LABEL_44:
                      sub_16CCBA0(v76, v52);
                      if ( v50 )
                        goto LABEL_55;
                    }
                    else
                    {
                      v56 = 0;
                      while ( v52 != *v53 )
                      {
                        if ( *v53 == -2 )
                        {
                          v56 = v53;
                          if ( v53 + 1 == v54 )
                            goto LABEL_54;
                          ++v53;
                        }
                        else if ( v54 == ++v53 )
                        {
                          if ( !v56 )
                            goto LABEL_64;
LABEL_54:
                          *v56 = v52;
                          --*(_DWORD *)(v51 + 32);
                          ++*(_QWORD *)v51;
                          goto LABEL_55;
                        }
                      }
                    }
                  }
                  v78 -= 24LL;
                  v29 = v77;
                  v47 = v78;
                  if ( v78 != v77 )
                    continue;
                  break;
                }
LABEL_67:
                v33 = v29;
              }
LABEL_56:
              if ( v29 )
                j_j___libc_free_0(v29, v79 - v29);
              ++v69;
              v21 = v86;
              v20 = v85;
              if ( v66 == v69 )
              {
                if ( v86 != v85 )
                  _libc_free((unsigned __int64)v86);
                return;
              }
            }
          }
        }
      }
    }
  }
}
