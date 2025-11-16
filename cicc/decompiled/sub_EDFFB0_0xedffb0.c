// Function: sub_EDFFB0
// Address: 0xedffb0
//
__int64 *__fastcall sub_EDFFB0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r15
  _QWORD *v3; // r14
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v7; // rsi
  char v8; // dl
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rax
  _DWORD *v12; // rax
  _DWORD *v13; // rdx
  int v14; // r12d
  __int64 *v15; // r13
  char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  char *v19; // rax
  __int64 (__fastcall *v20)(_QWORD *); // rax
  __int64 v21; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // r14
  _QWORD *v24; // r13
  _QWORD *v25; // rdi
  _QWORD *v26; // rbx
  _QWORD *v27; // r15
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r8
  unsigned __int64 *v32; // rbx
  unsigned __int64 v33; // rcx
  _QWORD *v34; // rax
  _QWORD *v35; // r12
  __int64 *v36; // r14
  __int64 v37; // r13
  __int64 *v38; // r15
  __int64 *v39; // rax
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // r15
  __int64 v42; // r13
  _BYTE *v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rax
  _QWORD *v47; // r12
  _QWORD *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 v51; // rax
  __int64 v52; // rdi
  int v53; // edx
  unsigned __int64 v54; // rdx
  char *v55; // rbx
  unsigned __int64 v56; // rcx
  unsigned __int64 v57; // r9
  int v58; // esi
  _QWORD *v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // rcx
  __int64 v62; // rcx
  __int64 v63; // rdi
  _QWORD *v64; // rdx
  _QWORD *v65; // rax
  __int64 v66; // rdi
  char *v67; // rbx
  __int64 v69; // [rsp+10h] [rbp-100h]
  unsigned __int64 v70; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v71; // [rsp+20h] [rbp-F0h]
  __int64 v72; // [rsp+28h] [rbp-E8h]
  __int64 *v73; // [rsp+28h] [rbp-E8h]
  __int64 v74; // [rsp+30h] [rbp-E0h]
  __int64 v75; // [rsp+30h] [rbp-E0h]
  int v76; // [rsp+30h] [rbp-E0h]
  _QWORD *v77; // [rsp+38h] [rbp-D8h]
  _QWORD *v78; // [rsp+40h] [rbp-D0h]
  __int64 v79; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE *v80; // [rsp+58h] [rbp-B8h]
  _BYTE *v81; // [rsp+60h] [rbp-B0h]
  __int64 v82; // [rsp+68h] [rbp-A8h]
  __int64 v83[4]; // [rsp+70h] [rbp-A0h] BYREF
  __m128i v84; // [rsp+90h] [rbp-80h] BYREF
  __int64 v85; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-68h]
  __int64 v87; // [rsp+B0h] [rbp-60h]
  __int64 v88; // [rsp+B8h] [rbp-58h]
  __int64 v89; // [rsp+C0h] [rbp-50h]
  __int64 v90; // [rsp+C8h] [rbp-48h]
  __int64 v91; // [rsp+D0h] [rbp-40h]
  char v92; // [rsp+D8h] [rbp-38h]

  v2 = a1;
  v3 = (_QWORD *)a2;
  v4 = *(_QWORD *)(a2 + 112);
  v5 = *(__int64 **)(v4 + 8);
  if ( (__int64)(*(_QWORD *)(v4 + 16) - (_QWORD)v5) <= 23 )
  {
    sub_ED6550(v84.m128i_i64, byte_3F871B3);
    sub_ED85B0(a1, a2, 8, &v84);
    if ( (__int64 *)v84.m128i_i64[0] != &v85 )
      j_j___libc_free_0(v84.m128i_i64[0], v85 + 1);
    return v2;
  }
  v7 = *(_QWORD *)(v4 + 8);
  sub_ED2E50(&v84, v5);
  v8 = v92 & 1;
  v9 = (2 * (v92 & 1)) | v92 & 0xFD;
  v92 = v9;
  if ( !v8 )
  {
    v11 = sub_ED2FC0((__int64)&v84);
    v12 = sub_ED97F0((__int64)v3, v84.m128i_i32[2], (__int64 *)((char *)v5 + v11), 0);
    v13 = v12;
    if ( (v84.m128i_i64[1] & 0x200000000000000LL) != 0 )
      v13 = sub_ED97F0((__int64)v3, v84.m128i_i32[2], v12, 1);
    v14 = v86;
    if ( (_DWORD)v86 )
    {
      sub_ED6550(v83, byte_3F871B3);
      v7 = (__int64)v3;
      sub_ED85B0(a1, (__int64)v3, 6, v83);
      sub_2240A30(v83);
      goto LABEL_7;
    }
    v72 = (__int64)v13;
    v74 = v84.m128i_i64[1];
    v15 = (__int64 *)((char *)v5 + v87);
    v78 = (_QWORD *)sub_22077B0(56);
    if ( v78 )
      sub_ED9D40(v78, v15, v72, (__int64)v5, 0, v74);
    if ( (unsigned __int64)sub_ED2E40((__int64)&v84) > 7 && (v84.m128i_i8[15] & 0x40) != 0 )
    {
      sub_EDE240(v83, v3 + 20, (__int64)v5, v88);
      v51 = v83[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v83[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_92;
      v83[0] = 0;
      sub_9C66B0(v83);
    }
    if ( (unsigned __int64)sub_ED2E40((__int64)&v84) > 8 )
    {
      v16 = (char *)v5 + v89;
      v17 = *(__int64 *)((char *)v5 + v89);
      if ( (v17 & 7) != 0 )
      {
        sub_ED6550(v83, byte_3F871B3);
        v53 = 4;
LABEL_78:
        v7 = (__int64)v3;
        sub_ED85B0(v2, (__int64)v3, v53, v83);
        sub_2240A30(v83);
        goto LABEL_21;
      }
      v3[60] = v17;
      v18 = v3[14];
      v19 = v16 + 8;
      v3[59] = v19;
      if ( *(_QWORD *)(v18 + 16) < (unsigned __int64)v19 )
      {
        v7 = (__int64)v83;
        LODWORD(v83[0]) = 9;
        sub_ED8730(a1, (int *)v83, "corrupted binary ids");
        goto LABEL_21;
      }
    }
    if ( (unsigned __int64)sub_ED2E40((__int64)&v84) > 0xB )
    {
      v29 = (unsigned __int64)v5 + v91 + 8;
      if ( v29 > *(_QWORD *)(v3[14] + 16LL) )
      {
        v7 = (__int64)v83;
        LODWORD(v83[0]) = 8;
        sub_ED8A30(a1, (int *)v83);
LABEL_21:
        if ( v78 )
        {
          v20 = *(__int64 (__fastcall **)(_QWORD *))(*v78 + 8LL);
          if ( v20 == sub_ED7520 )
          {
            *v78 = &unk_49E4D18;
            v21 = v78[1];
            v75 = v21;
            if ( v21 )
            {
              v22 = *(_QWORD **)(v21 + 32);
              v77 = *(_QWORD **)(v21 + 40);
              if ( v77 != v22 )
              {
                v73 = v2;
                do
                {
                  v23 = (_QWORD *)v22[6];
                  if ( v23 )
                  {
                    v24 = v23 + 9;
                    do
                    {
                      v25 = (_QWORD *)*(v24 - 3);
                      v26 = (_QWORD *)*(v24 - 2);
                      v24 -= 3;
                      v27 = v25;
                      if ( v26 != v25 )
                      {
                        do
                        {
                          if ( *v27 )
                            j_j___libc_free_0(*v27, v27[2] - *v27);
                          v27 += 3;
                        }
                        while ( v26 != v27 );
                        v25 = (_QWORD *)*v24;
                      }
                      if ( v25 )
                        j_j___libc_free_0(v25, v24[2] - (_QWORD)v25);
                    }
                    while ( v23 != v24 );
                    j_j___libc_free_0(v23, 72);
                  }
                  v28 = v22[3];
                  if ( v28 )
                    j_j___libc_free_0(v28, v22[5] - v28);
                  if ( *v22 )
                    j_j___libc_free_0(*v22, v22[2] - *v22);
                  v22 += 10;
                }
                while ( v77 != v22 );
                v2 = v73;
                v22 = *(_QWORD **)(v75 + 32);
              }
              if ( v22 )
                j_j___libc_free_0(v22, *(_QWORD *)(v75 + 48) - (_QWORD)v22);
              j_j___libc_free_0(v75, 80);
            }
            v7 = 56;
            j_j___libc_free_0(v78, 56);
          }
          else
          {
            v20(v78);
          }
        }
        goto LABEL_7;
      }
      v30 = *(__int64 *)((char *)v5 + v91);
      v3[57] = v29;
      v3[58] = v30;
    }
    if ( (unsigned __int64)sub_ED2E40((__int64)&v84) > 9 && v84.m128i_i64[1] < 0 )
    {
      v32 = (unsigned __int64 *)((char *)v5 + v90);
      v71 = *(_QWORD *)(v3[14] + 16LL);
      if ( v71 < (unsigned __int64)(v32 + 2) )
      {
LABEL_105:
        sub_ED6550(v83, byte_3F871B3);
        v53 = 8;
        goto LABEL_78;
      }
      v70 = *v32;
      v33 = *v32;
      v3[13] = v32[1];
      if ( v33 )
      {
        if ( v71 >= (unsigned __int64)(v32 + 4) )
        {
          v82 = 1;
          v79 = 0;
          v80 = 0;
          v81 = 0;
          v82 = v32[2];
          if ( v71 < (unsigned __int64)&v32[v32[3] + 4] )
          {
LABEL_89:
            sub_ED6550(v83, byte_3F871B3);
            v7 = (__int64)v3;
            sub_ED85B0(v2, (__int64)v3, 8, v83);
            sub_2240A30(v83);
            if ( v79 )
            {
              v7 = (__int64)&v81[-v79];
              j_j___libc_free_0(v79, &v81[-v79]);
            }
            goto LABEL_21;
          }
          v34 = v3 + 7;
          v76 = v14;
          v35 = v3;
          v36 = (__int64 *)(v32 + 4);
          v37 = v32[3];
          v69 = (__int64)v34;
          v38 = &v79;
          while ( 1 )
          {
            if ( v37 )
            {
              v39 = v38;
              LODWORD(v40) = 0;
              v41 = v37;
              v42 = (__int64)v39;
              do
              {
                v44 = *v36;
                v43 = v80;
                ++v36;
                v83[0] = v44;
                if ( v80 == v81 )
                {
                  sub_9CA200(v42, v80, v83);
                }
                else
                {
                  if ( v80 )
                  {
                    *(_QWORD *)v80 = v44;
                    v43 = v80;
                  }
                  v80 = v43 + 8;
                }
                v40 = (unsigned int)(v40 + 1);
              }
              while ( v40 < v41 );
              v38 = (__int64 *)v42;
            }
            v54 = *((unsigned int *)v35 + 16);
            v55 = (char *)v38;
            v56 = v35[7];
            v57 = v54 + 1;
            v58 = *((_DWORD *)v35 + 16);
            if ( v54 + 1 > *((unsigned int *)v35 + 17) )
            {
              if ( v56 > (unsigned __int64)v38 || (v54 = v56 + 32 * v54, (unsigned __int64)v38 >= v54) )
              {
                sub_EDFEB0(v69, v57, v54, v56, v31, v57);
                v54 = *((unsigned int *)v35 + 16);
                v56 = v35[7];
                v58 = *((_DWORD *)v35 + 16);
              }
              else
              {
                v67 = (char *)v38 - v56;
                sub_EDFEB0(v69, v57, v54, v56, v31, v57);
                v56 = v35[7];
                v54 = *((unsigned int *)v35 + 16);
                v55 = &v67[v56];
                v58 = *((_DWORD *)v35 + 16);
              }
            }
            v59 = (_QWORD *)(v56 + 32 * v54);
            if ( v59 )
            {
              v60 = *(_QWORD *)v55;
              *(_QWORD *)v55 = 0;
              *v59 = v60;
              v61 = *((_QWORD *)v55 + 1);
              *((_QWORD *)v55 + 1) = 0;
              v59[1] = v61;
              v62 = *((_QWORD *)v55 + 2);
              *((_QWORD *)v55 + 2) = 0;
              v59[2] = v62;
              v59[3] = *((_QWORD *)v55 + 3);
              v58 = *((_DWORD *)v35 + 16);
            }
            v63 = v79;
            *((_DWORD *)v35 + 16) = v58 + 1;
            if ( v63 )
              j_j___libc_free_0(v63, &v81[-v63]);
            if ( (unsigned int)++v76 >= v70 )
            {
              v2 = a1;
              v3 = v35;
              goto LABEL_67;
            }
            v64 = v36 + 2;
            if ( v71 < (unsigned __int64)(v36 + 2) )
              break;
            v82 = 1;
            v79 = 0;
            v80 = 0;
            v81 = 0;
            v82 = *v36;
            v37 = v36[1];
            v36 += 2;
            if ( v71 < (unsigned __int64)&v64[v37] )
            {
              v2 = a1;
              v3 = v35;
              goto LABEL_89;
            }
          }
          v2 = a1;
          v3 = v35;
        }
        goto LABEL_105;
      }
    }
LABEL_67:
    v45 = v3[15];
    if ( !v45 )
    {
      v65 = (_QWORD *)sub_22077B0(16);
      if ( v65 )
      {
        *v65 = off_49E4CB8;
        v65[1] = v78;
      }
      v66 = v3[17];
      v3[17] = v65;
      if ( v66 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v66 + 8LL))(v66);
      goto LABEL_74;
    }
    v3[15] = 0;
    v83[0] = v45;
    v46 = (_QWORD *)sub_22077B0(64);
    v47 = v46;
    if ( v46 )
    {
      v48 = v46 + 2;
      *v46 = &unk_49E4DB8;
      v49 = v83[0];
      v83[0] = 0;
      v47[1] = v49;
      sub_EE5C10(v48);
      v47[3] = 0;
      v47[4] = 0;
      v47[5] = 0;
      *((_DWORD *)v47 + 12) = 0;
      v47[7] = v78;
    }
    sub_C21E00(v83);
    v50 = v3[17];
    v3[17] = v47;
    if ( v50 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v50 + 8LL))(v50);
      v47 = (_QWORD *)v3[17];
    }
    (*(void (__fastcall **)(__int64 *, _QWORD *))(*v47 + 16LL))(v83, v47);
    v51 = v83[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v83[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v83[0] = 0;
      sub_9C66B0(v83);
LABEL_74:
      v52 = v3[16];
      v3[16] = v78;
      if ( v52 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v52 + 8LL))(v52);
      v7 = (__int64)v3;
      sub_ED8620(v2, (__int64)v3);
      goto LABEL_7;
    }
LABEL_92:
    v7 = (__int64)v83;
    v83[0] = v51 | 1;
    *v2 = 0;
    sub_9C6670(v2, v83);
    sub_9C66B0(v83);
    goto LABEL_21;
  }
  v83[0] = 0;
  v92 = v9 & 0xFD;
  v10 = v84.m128i_i64[0];
  v84.m128i_i64[0] = 0;
  *a1 = v10 | 1;
  sub_9C8CB0(v83);
LABEL_7:
  if ( (v92 & 2) != 0 )
    sub_EDE400(&v84, v7);
  if ( (v92 & 1) != 0 )
    sub_9C8CB0(v84.m128i_i64);
  return v2;
}
