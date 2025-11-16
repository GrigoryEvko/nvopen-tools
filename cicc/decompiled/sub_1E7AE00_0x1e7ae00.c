// Function: sub_1E7AE00
// Address: 0x1e7ae00
//
__int64 __fastcall sub_1E7AE00(_QWORD *a1, __int64 a2, unsigned __int64 a3, _BYTE *a4, _QWORD *a5)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v10; // rax
  int v11; // r12d
  unsigned __int8 v12; // al
  char v13; // dl
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  int v20; // edx
  _BYTE *v21; // rax
  _BYTE *v22; // r8
  char *v23; // rdi
  size_t v24; // r9
  __int64 v25; // r10
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // esi
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 **v34; // r9
  __int64 **v35; // r8
  __int64 v36; // rax
  _QWORD *v37; // r14
  __int64 **v38; // r12
  __int64 v39; // r13
  __int64 **v40; // rbx
  int v41; // r8d
  int v42; // r9d
  __int64 v43; // r11
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 **v48; // rsi
  __int64 v49; // r14
  int v50; // r13d
  __int64 v51; // r12
  __int64 *v52; // rbx
  __int64 v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // r11
  unsigned int v57; // r12d
  __int64 v58; // rax
  __int64 *v59; // r10
  __int64 v60; // rax
  __int64 *v61; // r11
  __int64 v62; // rbx
  __int64 *v63; // r14
  __int64 *v64; // r15
  char *v65; // rax
  __int64 v66; // rcx
  __int64 *v67; // rsi
  __int64 *v68; // rdi
  char *v69; // rax
  __int64 v70; // r9
  int v71; // r9d
  _QWORD *v72; // rax
  _BYTE *v73; // rdi
  size_t v74; // rdx
  int v75; // eax
  int v76; // r9d
  __int64 v77; // rax
  size_t v78; // [rsp+0h] [rbp-110h]
  __int64 v79; // [rsp+0h] [rbp-110h]
  _BYTE *v80; // [rsp+8h] [rbp-108h]
  __int64 v81; // [rsp+8h] [rbp-108h]
  __int64 v82; // [rsp+8h] [rbp-108h]
  __int64 v83; // [rsp+10h] [rbp-100h]
  _BYTE *v84; // [rsp+10h] [rbp-100h]
  __int64 v85; // [rsp+10h] [rbp-100h]
  unsigned int v86; // [rsp+1Ch] [rbp-F4h]
  _QWORD *v87; // [rsp+20h] [rbp-F0h]
  int v88; // [rsp+28h] [rbp-E8h]
  __int64 v89; // [rsp+28h] [rbp-E8h]
  int v90; // [rsp+28h] [rbp-E8h]
  __int64 v91; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v92; // [rsp+28h] [rbp-E8h]
  int v93; // [rsp+28h] [rbp-E8h]
  __int64 v94; // [rsp+30h] [rbp-E0h]
  __int64 v95; // [rsp+30h] [rbp-E0h]
  char *v96; // [rsp+30h] [rbp-E0h]
  _QWORD *v97; // [rsp+30h] [rbp-E0h]
  int v98; // [rsp+30h] [rbp-E0h]
  __int64 v99; // [rsp+38h] [rbp-D8h]
  __int64 **v100; // [rsp+38h] [rbp-D8h]
  __int64 v103; // [rsp+58h] [rbp-B8h]
  __int64 v104; // [rsp+58h] [rbp-B8h]
  __int64 v106; // [rsp+68h] [rbp-A8h]
  void *src; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v108; // [rsp+78h] [rbp-98h]
  _BYTE dest[32]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int64 v110; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE *v111; // [rsp+A8h] [rbp-68h] BYREF
  __int64 v112; // [rsp+B0h] [rbp-60h]
  _BYTE v113[88]; // [rsp+B8h] [rbp-58h] BYREF

  v5 = *(unsigned int *)(a2 + 40);
  if ( !(_DWORD)v5 )
    return 0;
  v6 = a2;
  v7 = 0;
  v8 = 40 * v5;
  v106 = 0;
  v86 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v87 = a5 + 1;
  do
  {
    while ( 1 )
    {
      v10 = v7 + *(_QWORD *)(v6 + 32);
      if ( *(_BYTE *)v10 )
        goto LABEL_5;
      v11 = *(_DWORD *)(v10 + 8);
      if ( !v11 )
        goto LABEL_5;
      v12 = *(_BYTE *)(v10 + 3);
      v13 = v12 & 0x10;
      if ( v11 > 0 )
        break;
      if ( v13 )
      {
        v14 = a1[29];
        v103 = 16LL * (v11 & 0x7FFFFFFF);
        v15 = *(__int64 (**)())(*(_QWORD *)v14 + 728LL);
        if ( v15 != sub_1E77D50
          && !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v15)(
                v14,
                *(_QWORD *)(*(_QWORD *)(a1[31] + 24LL) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
        {
          return 0;
        }
        if ( v106 )
        {
          v45 = a1[31];
          LOBYTE(v110) = 0;
          v46 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + v103 + 8);
          if ( v46 )
          {
            while ( (*(_BYTE *)(v46 + 3) & 0x10) != 0 || (*(_BYTE *)(v46 + 4) & 8) != 0 )
            {
              v46 = *(_QWORD *)(v46 + 32);
              if ( !v46 )
                goto LABEL_5;
            }
            if ( !(unsigned __int8)sub_1E78170((__int64)a1, v11, v106, a3, a4, &v110) )
              return 0;
          }
        }
        else
        {
          v16 = (_QWORD *)a5[2];
          if ( !v16 )
            goto LABEL_19;
          v17 = v87;
          do
          {
            while ( 1 )
            {
              v18 = v16[2];
              v19 = v16[3];
              if ( v16[4] >= a3 )
                break;
              v16 = (_QWORD *)v16[3];
              if ( !v19 )
                goto LABEL_17;
            }
            v17 = v16;
            v16 = (_QWORD *)v16[2];
          }
          while ( v18 );
LABEL_17:
          if ( v87 != v17 && v17[4] <= a3 )
          {
            v48 = (__int64 **)(v17 + 5);
          }
          else
          {
LABEL_19:
            v20 = 0;
            v21 = *(_BYTE **)(a3 + 96);
            v22 = *(_BYTE **)(a3 + 88);
            v23 = dest;
            v108 = 0x400000000LL;
            src = dest;
            v24 = v21 - v22;
            v25 = (v21 - v22) >> 3;
            if ( (unsigned __int64)(v21 - v22) > 0x20 )
            {
              v78 = v21 - v22;
              v80 = v22;
              v84 = v21;
              v92 = (v21 - v22) >> 3;
              sub_16CD150((__int64)&src, dest, v92, 8, (int)v22, v24);
              v20 = v108;
              v24 = v78;
              v22 = v80;
              v21 = v84;
              LODWORD(v25) = v92;
              v23 = (char *)src + 8 * (unsigned int)v108;
            }
            if ( v21 != v22 )
            {
              v88 = v25;
              memmove(v23, v22, v24);
              v20 = v108;
              LODWORD(v25) = v88;
            }
            v26 = a1[32];
            LODWORD(v108) = v25 + v20;
            v89 = v26;
            sub_1E06620(v26);
            v27 = *(_QWORD *)(v89 + 1312);
            v28 = *(unsigned int *)(v27 + 48);
            if ( !(_DWORD)v28 )
              goto LABEL_87;
            v29 = *(_QWORD *)(v27 + 32);
            v30 = (v28 - 1) & v86;
            v31 = (__int64 *)(v29 + 16LL * v30);
            v32 = *v31;
            if ( a3 != *v31 )
            {
              v75 = 1;
              while ( v32 != -8 )
              {
                v76 = v75 + 1;
                v77 = ((_DWORD)v28 - 1) & (v30 + v75);
                v30 = v77;
                v31 = (__int64 *)(v29 + 16 * v77);
                v32 = *v31;
                if ( a3 == *v31 )
                  goto LABEL_25;
                v75 = v76;
              }
LABEL_87:
              BUG();
            }
LABEL_25:
            if ( v31 == (__int64 *)(v29 + 16 * v28) )
              goto LABEL_87;
            v33 = v31[1];
            v34 = *(__int64 ***)(v33 + 32);
            v35 = *(__int64 ***)(v33 + 24);
            if ( v35 != v34 )
            {
              v36 = v6;
              v90 = v11;
              v37 = a1;
              v38 = v35;
              v83 = v7;
              v39 = v36;
              v40 = v34;
              do
              {
                if ( *(_QWORD *)(*v38)[1] == *(_QWORD *)(v39 + 24) && !sub_1DD6970(a3, **v38) )
                {
                  v43 = **v38;
                  v44 = (unsigned int)v108;
                  if ( (unsigned int)v108 >= HIDWORD(v108) )
                  {
                    v82 = **v38;
                    sub_16CD150((__int64)&src, dest, 0, 8, v41, v42);
                    v44 = (unsigned int)v108;
                    v43 = v82;
                  }
                  *((_QWORD *)src + v44) = v43;
                  LODWORD(v108) = v108 + 1;
                }
                ++v38;
              }
              while ( v40 != v38 );
              v58 = v39;
              v11 = v90;
              a1 = v37;
              v7 = v83;
              v6 = v58;
            }
            v59 = (__int64 *)src;
            v60 = 8LL * (unsigned int)v108;
            v61 = (__int64 *)((char *)src + v60);
            if ( v60 )
            {
              v95 = v7;
              v62 = v60 >> 3;
              v85 = v6;
              v63 = (__int64 *)src;
              v81 = v8;
              v64 = (__int64 *)((char *)src + v60);
              do
              {
                v65 = (char *)sub_2207800(8 * v62, &unk_435FF63);
                if ( v65 )
                {
                  v79 = 8 * v62;
                  v66 = v62;
                  v67 = v64;
                  v68 = v63;
                  v7 = v95;
                  v6 = v85;
                  v96 = v65;
                  v8 = v81;
                  sub_1E79E80(v68, v67, v65, v66, (__int64)a1);
                  v69 = v96;
                  v70 = v79;
                  goto LABEL_70;
                }
                v62 >>= 1;
              }
              while ( v62 );
              v59 = v63;
              v61 = v64;
              v7 = v95;
              v6 = v85;
              v8 = v81;
            }
            sub_1E794C0(v59, v61, (__int64)a1);
            v69 = 0;
            v70 = 0;
LABEL_70:
            j_j___libc_free_0(v69, v70);
            v71 = v108;
            v111 = v113;
            v110 = a3;
            v112 = 0x400000000LL;
            if ( (_DWORD)v108 )
            {
              v73 = v113;
              v74 = 8LL * (unsigned int)v108;
              if ( (unsigned int)v108 <= 4
                || (v93 = v108,
                    sub_16CD150((__int64)&v111, v113, (unsigned int)v108, 8, (int)v113, v108),
                    v73 = v111,
                    v71 = v93,
                    (v74 = 8LL * (unsigned int)v108) != 0) )
              {
                v98 = v71;
                memcpy(v73, src, v74);
                v71 = v98;
              }
              LODWORD(v112) = v71;
            }
            v72 = sub_1E7AC00(a5, (__int64)&v110);
            if ( v111 != v113 )
            {
              v97 = v72;
              _libc_free((unsigned __int64)v111);
              v72 = v97;
            }
            v48 = (__int64 **)(v72 + 5);
            if ( src != dest )
            {
              v100 = (__int64 **)(v72 + 5);
              _libc_free((unsigned __int64)src);
              v48 = v100;
            }
          }
          v99 = (__int64)&(*v48)[*((unsigned int *)v48 + 2)];
          if ( *v48 == (__int64 *)v99 )
            return 0;
          v91 = v6;
          v49 = (__int64)a1;
          v50 = v11;
          v51 = v103;
          v104 = v8;
          v94 = v7;
          v52 = *v48;
          while ( 1 )
          {
            v53 = *(_QWORD *)(v49 + 248);
            v54 = *v52;
            LOBYTE(v110) = 0;
            v55 = *(_QWORD *)(*(_QWORD *)(v53 + 24) + v51 + 8);
            if ( !v55 )
              break;
            while ( (*(_BYTE *)(v55 + 3) & 0x10) != 0 || (*(_BYTE *)(v55 + 4) & 8) != 0 )
            {
              v55 = *(_QWORD *)(v55 + 32);
              if ( !v55 )
                goto LABEL_62;
            }
            if ( (unsigned __int8)sub_1E78170(v49, v50, v54, a3, a4, &v110) )
              break;
            if ( (_BYTE)v110 )
              return v106;
            if ( (__int64 *)v99 == ++v52 )
              return 0;
          }
LABEL_62:
          v56 = v54;
          v57 = v50;
          a1 = (_QWORD *)v49;
          v7 = v94;
          v6 = v91;
          v8 = v104;
          if ( !v56 )
            return 0;
          v106 = v56;
          if ( !(unsigned __int8)sub_1E7B680(a1, v57, v91, a3, v56, a5) )
            return 0;
        }
      }
LABEL_5:
      v7 += 40;
      if ( v8 == v7 )
        goto LABEL_44;
    }
    if ( v13 )
    {
      if ( (((v12 & 0x10) != 0) & (v12 >> 6)) == 0 )
        return 0;
      goto LABEL_5;
    }
    if ( !(unsigned __int8)sub_1E69FD0((_QWORD *)a1[31], v11) )
      return 0;
    v7 += 40;
  }
  while ( v8 != v7 );
LABEL_44:
  if ( a3 == v106 || !v106 || *(_BYTE *)(v106 + 180) )
    return 0;
  return v106;
}
