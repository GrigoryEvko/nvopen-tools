// Function: sub_1AA92B0
// Address: 0x1aa92b0
//
_QWORD *__fastcall sub_1AA92B0(
        __int64 a1,
        __int64 a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // rsi
  __int64 *v16; // rsi
  _QWORD *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r12
  _QWORD *v21; // rax
  __int64 v22; // r15
  __int64 v23; // rsi
  __int64 *v24; // r12
  _QWORD *v25; // rax
  _QWORD *v26; // r12
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // r12
  __int64 v30; // rbx
  int v31; // ecx
  unsigned int v32; // esi
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // rax
  _BYTE *v36; // rdx
  _BYTE *v37; // rsi
  __int64 *v38; // rax
  unsigned int v39; // esi
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // r12
  __int64 v43; // rbx
  _BYTE *v44; // rsi
  __int64 *v45; // rax
  __int64 v46; // r8
  __int64 *v47; // r12
  __int64 v48; // rdi
  __int64 v49; // r12
  __int64 v50; // rdi
  __int64 *v51; // r13
  _BYTE *v52; // rsi
  __int64 v53; // r12
  __int64 v54; // rax
  _BYTE *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  int v58; // r8d
  int v59; // r9d
  __int64 v60; // rax
  unsigned int v61; // edi
  __int64 v62; // rcx
  _QWORD *v63; // rdx
  _QWORD *v64; // rsi
  __int64 v65; // rbx
  __int64 v66; // r12
  _BYTE *v67; // rsi
  __int64 *v68; // rax
  __int64 v69; // r14
  __int64 v70; // rdi
  __int64 v71; // r12
  __int64 v72; // rdi
  int v73; // eax
  int v74; // esi
  __int64 v75; // rcx
  unsigned int v76; // edx
  _QWORD *v77; // rax
  _QWORD *v78; // rdi
  __int64 v79; // r12
  __int64 v81; // rsi
  unsigned __int8 *v82; // rsi
  _QWORD *v83; // rax
  int v84; // eax
  int v85; // r8d
  int v86; // eax
  int v87; // edi
  int v88; // edx
  int v89; // eax
  int v90; // r9d
  int v91; // edi
  _QWORD *v92; // [rsp+10h] [rbp-C0h]
  __int64 v93; // [rsp+18h] [rbp-B8h]
  _BYTE *v94; // [rsp+20h] [rbp-B0h]
  __int64 v95; // [rsp+30h] [rbp-A0h]
  unsigned int v96; // [rsp+30h] [rbp-A0h]
  __int64 *v97; // [rsp+38h] [rbp-98h]
  __int64 *desta; // [rsp+40h] [rbp-90h]
  size_t n; // [rsp+48h] [rbp-88h]
  size_t na; // [rsp+48h] [rbp-88h]
  _QWORD *v104; // [rsp+58h] [rbp-78h]
  __int64 v105; // [rsp+60h] [rbp-70h]
  __int64 v107; // [rsp+70h] [rbp-60h] BYREF
  __int64 v108; // [rsp+78h] [rbp-58h] BYREF
  __int64 v109[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v110; // [rsp+90h] [rbp-40h]

  v16 = (__int64 *)(a2 + 24);
  v17 = (_QWORD *)v16[2];
  v110 = 257;
  v105 = sub_157FBF0(v17, v16, (__int64)v109);
  n = sub_157EBA0((__int64)v17);
  v18 = sub_157E9C0((__int64)v17);
  v19 = v17[7];
  v110 = 257;
  v20 = v18;
  v95 = v19;
  v21 = (_QWORD *)sub_22077B0(64);
  v22 = (__int64)v21;
  if ( v21 )
    sub_157FB60(v21, v20, (__int64)v109, v95, v105);
  if ( a3 )
  {
    v104 = sub_1648A60(56, 0);
    if ( v104 )
      sub_15F82E0((__int64)v104, v20, v22);
  }
  else
  {
    v83 = sub_1648A60(56, 1u);
    v104 = v83;
    if ( v83 )
      sub_15F8590((__int64)v83, v105, v22);
  }
  v23 = *(_QWORD *)(a2 + 48);
  v109[0] = v23;
  v24 = v104 + 6;
  if ( !v23 )
  {
    if ( v24 == v109 )
      goto LABEL_10;
    v81 = v104[6];
    if ( !v81 )
      goto LABEL_10;
LABEL_73:
    sub_161E7C0((__int64)v24, v81);
    goto LABEL_74;
  }
  sub_1623A60((__int64)v109, v23, 2);
  if ( v24 == v109 )
  {
    if ( v109[0] )
      sub_161E7C0((__int64)v109, v109[0]);
    goto LABEL_10;
  }
  v81 = v104[6];
  if ( v81 )
    goto LABEL_73;
LABEL_74:
  v82 = (unsigned __int8 *)v109[0];
  v104[6] = v109[0];
  if ( v82 )
    sub_1623210((__int64)v109, v82, (__int64)v24);
LABEL_10:
  v25 = sub_1648A60(56, 3u);
  v26 = v25;
  if ( v25 )
    sub_15F83E0((__int64)v25, v22, v105, a1, 0);
  sub_1625C10((__int64)v26, 2, a4);
  sub_1AA6530(n, v26, a7, a8, a9, a10, v27, v28, a13, a14);
  if ( a5 )
  {
    v29 = *(unsigned int *)(a5 + 48);
    if ( (_DWORD)v29 )
    {
      v30 = *(_QWORD *)(a5 + 32);
      v31 = v29 - 1;
      v96 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
      v32 = (v29 - 1) & v96;
      v33 = (_QWORD *)(v30 + 16LL * v32);
      v34 = (_QWORD *)*v33;
      if ( v17 == (_QWORD *)*v33 )
      {
LABEL_15:
        if ( v33 == (_QWORD *)(v30 + 16LL * (unsigned int)v29) )
          goto LABEL_66;
        v35 = v33[1];
        if ( !v35 )
          goto LABEL_66;
        v36 = *(_BYTE **)(v35 + 32);
        v37 = *(_BYTE **)(v35 + 24);
        na = v36 - v37;
        if ( (unsigned __int64)(v36 - v37) > 0x7FFFFFFFFFFFFFF8LL )
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        if ( na )
        {
          v94 = *(_BYTE **)(v35 + 32);
          v38 = (__int64 *)sub_22077B0(na);
          desta = v38;
          v30 = *(_QWORD *)(a5 + 32);
          v97 = (__int64 *)((char *)v38 + na);
          v29 = *(unsigned int *)(a5 + 48);
          if ( v37 != v94 )
            memcpy(v38, v37, na);
        }
        else
        {
          if ( v37 == v36 )
          {
            v97 = 0;
            desta = 0;
            v107 = v105;
            goto LABEL_23;
          }
          v97 = 0;
          desta = 0;
          memcpy(0, v37, 0);
        }
        v107 = v105;
        if ( !(_DWORD)v29 )
          goto LABEL_106;
        v31 = v29 - 1;
LABEL_23:
        v39 = v31 & v96;
        v40 = (_QWORD *)(v30 + 16LL * (v31 & v96));
        v41 = (_QWORD *)*v40;
        if ( v17 == (_QWORD *)*v40 )
        {
LABEL_24:
          if ( v40 != (_QWORD *)(v30 + 16 * v29) )
          {
            v42 = v40[1];
            *(_BYTE *)(a5 + 72) = 0;
            sub_1AA56D0(v109, v105, v42);
            v43 = v109[0];
            v108 = v109[0];
            v44 = *(_BYTE **)(v42 + 32);
            if ( v44 == *(_BYTE **)(v42 + 40) )
            {
              sub_15CE310(v42 + 24, v44, &v108);
              v43 = v109[0];
            }
            else
            {
              if ( v44 )
              {
                *(_QWORD *)v44 = v109[0];
                v44 = *(_BYTE **)(v42 + 32);
                v43 = v109[0];
              }
              *(_QWORD *)(v42 + 32) = v44 + 8;
            }
            v109[0] = 0;
            v45 = sub_15CFF10(a5 + 24, &v107);
            v46 = v45[1];
            v47 = v45;
            v45[1] = v43;
            if ( v46 )
            {
              v48 = *(_QWORD *)(v46 + 24);
              if ( v48 )
              {
                v93 = v46;
                j_j___libc_free_0(v48, *(_QWORD *)(v46 + 40) - v48);
                v46 = v93;
              }
              j_j___libc_free_0(v46, 56);
              v43 = v47[1];
            }
            v49 = v109[0];
            if ( v109[0] )
            {
              v50 = *(_QWORD *)(v109[0] + 24);
              if ( v50 )
                j_j___libc_free_0(v50, *(_QWORD *)(v109[0] + 40) - v50);
              j_j___libc_free_0(v49, 56);
            }
            if ( v97 != desta )
            {
              v92 = v17;
              v51 = desta;
              do
              {
                v53 = *v51;
                *(_BYTE *)(a5 + 72) = 0;
                v54 = *(_QWORD *)(v53 + 8);
                if ( v43 != v54 )
                {
                  v109[0] = v53;
                  v55 = sub_1AA5610(*(_QWORD **)(v54 + 24), *(_QWORD *)(v54 + 32), v109);
                  sub_15CDF70(*(_QWORD *)(v53 + 8) + 24LL, v55);
                  *(_QWORD *)(v53 + 8) = v43;
                  v109[0] = v53;
                  v52 = *(_BYTE **)(v43 + 32);
                  if ( v52 == *(_BYTE **)(v43 + 40) )
                  {
                    sub_15CE310(v43 + 24, v52, v109);
                  }
                  else
                  {
                    if ( v52 )
                    {
                      *(_QWORD *)v52 = v53;
                      v52 = *(_BYTE **)(v43 + 32);
                    }
                    v52 += 8;
                    *(_QWORD *)(v43 + 32) = v52;
                  }
                  if ( *(_DWORD *)(v53 + 16) != *(_DWORD *)(*(_QWORD *)(v53 + 8) + 16LL) + 1 )
                    sub_1AA5500(v53, (__int64)v52, v56, v57, v58, v59);
                }
                ++v51;
              }
              while ( v97 != v51 );
              v17 = v92;
            }
            v107 = v22;
            v60 = *(unsigned int *)(a5 + 48);
            if ( (_DWORD)v60 )
            {
              v61 = (v60 - 1) & v96;
              v62 = *(_QWORD *)(a5 + 32);
              v63 = (_QWORD *)(v62 + 16LL * v61);
              v64 = (_QWORD *)*v63;
              if ( v17 == (_QWORD *)*v63 )
              {
LABEL_51:
                if ( v63 != (_QWORD *)(v62 + 16 * v60) )
                {
                  v65 = v63[1];
                  *(_BYTE *)(a5 + 72) = 0;
                  sub_1AA56D0(v109, v22, v65);
                  v66 = v109[0];
                  v108 = v109[0];
                  v67 = *(_BYTE **)(v65 + 32);
                  if ( v67 == *(_BYTE **)(v65 + 40) )
                  {
                    sub_15CE310(v65 + 24, v67, &v108);
                    v66 = v109[0];
                  }
                  else
                  {
                    if ( v67 )
                    {
                      *(_QWORD *)v67 = v109[0];
                      v67 = *(_BYTE **)(v65 + 32);
                      v66 = v109[0];
                    }
                    *(_QWORD *)(v65 + 32) = v67 + 8;
                  }
                  v109[0] = 0;
                  v68 = sub_15CFF10(a5 + 24, &v107);
                  v69 = v68[1];
                  v68[1] = v66;
                  if ( v69 )
                  {
                    v70 = *(_QWORD *)(v69 + 24);
                    if ( v70 )
                      j_j___libc_free_0(v70, *(_QWORD *)(v69 + 40) - v70);
                    j_j___libc_free_0(v69, 56);
                  }
                  v71 = v109[0];
                  if ( v109[0] )
                  {
                    v72 = *(_QWORD *)(v109[0] + 24);
                    if ( v72 )
                      j_j___libc_free_0(v72, *(_QWORD *)(v109[0] + 40) - v72);
                    j_j___libc_free_0(v71, 56);
                  }
                  if ( desta )
                    j_j___libc_free_0(desta, na);
                  goto LABEL_66;
                }
              }
              else
              {
                v88 = 1;
                while ( v64 != (_QWORD *)-8LL )
                {
                  v90 = v88 + 1;
                  v61 = (v60 - 1) & (v88 + v61);
                  v63 = (_QWORD *)(v62 + 16LL * v61);
                  v64 = (_QWORD *)*v63;
                  if ( v17 == (_QWORD *)*v63 )
                    goto LABEL_51;
                  v88 = v90;
                }
              }
            }
            v14 = v22;
            *(_BYTE *)(a5 + 72) = 0;
LABEL_107:
            sub_1AA56D0(v109, v14, 0);
            v108 = v109[0];
            BUG();
          }
        }
        else
        {
          v89 = 1;
          while ( v41 != (_QWORD *)-8LL )
          {
            v91 = v89 + 1;
            v39 = v31 & (v89 + v39);
            v40 = (_QWORD *)(v30 + 16LL * v39);
            v41 = (_QWORD *)*v40;
            if ( v17 == (_QWORD *)*v40 )
              goto LABEL_24;
            v89 = v91;
          }
        }
LABEL_106:
        v14 = v105;
        *(_BYTE *)(a5 + 72) = 0;
        goto LABEL_107;
      }
      v86 = 1;
      while ( v34 != (_QWORD *)-8LL )
      {
        v87 = v86 + 1;
        v32 = v31 & (v86 + v32);
        v33 = (_QWORD *)(v30 + 16LL * v32);
        v34 = (_QWORD *)*v33;
        if ( v17 == (_QWORD *)*v33 )
          goto LABEL_15;
        v86 = v87;
      }
    }
  }
LABEL_66:
  if ( a6 )
  {
    v73 = *(_DWORD *)(a6 + 24);
    if ( v73 )
    {
      v74 = v73 - 1;
      v75 = *(_QWORD *)(a6 + 8);
      v76 = (v73 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v77 = (_QWORD *)(v75 + 16LL * v76);
      v78 = (_QWORD *)*v77;
      if ( v17 == (_QWORD *)*v77 )
      {
LABEL_69:
        v79 = v77[1];
        if ( v79 )
        {
          sub_1400330(v77[1], v22, a6);
          sub_1400330(v79, v105, a6);
        }
      }
      else
      {
        v84 = 1;
        while ( v78 != (_QWORD *)-8LL )
        {
          v85 = v84 + 1;
          v76 = v74 & (v84 + v76);
          v77 = (_QWORD *)(v75 + 16LL * v76);
          v78 = (_QWORD *)*v77;
          if ( v17 == (_QWORD *)*v77 )
            goto LABEL_69;
          v84 = v85;
        }
      }
    }
  }
  return v104;
}
