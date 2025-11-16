// Function: sub_1B34B70
// Address: 0x1b34b70
//
__int64 __fastcall sub_1B34B70(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
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
        __m128 a14,
        char a15)
{
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rdi
  _QWORD *v21; // r12
  double v22; // xmm4_8
  double v23; // xmm5_8
  char v24; // al
  int v25; // ebx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r14
  int v29; // r14d
  __int64 v30; // rax
  __int64 v31; // rdx
  int v32; // eax
  int v33; // ebx
  __int64 i; // rax
  _QWORD *v35; // rdx
  __int64 result; // rax
  unsigned __int64 v37; // rdx
  _QWORD **v38; // r13
  _QWORD *v39; // r12
  __int64 *v40; // rax
  __int64 v41; // rsi
  int v42; // eax
  __int64 v43; // rcx
  unsigned int v44; // edx
  _QWORD *v45; // rax
  _QWORD *v46; // rdi
  int v47; // eax
  int v48; // edi
  __int64 v49; // rcx
  __int64 v50; // rsi
  unsigned int v51; // edx
  __int64 *v52; // rax
  __int64 v53; // r8
  int v54; // edx
  int v55; // ecx
  __int64 v56; // rsi
  unsigned int v57; // edx
  _QWORD *v58; // rax
  _QWORD *v59; // rdi
  __int64 v60; // rdx
  int v61; // r8d
  int v62; // r9d
  __int64 v63; // r14
  int v64; // eax
  __int64 v65; // rcx
  int v66; // esi
  unsigned int v67; // edx
  _QWORD *v68; // rax
  _QWORD *v69; // rdi
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rdi
  unsigned __int64 v74; // rcx
  __int64 v75; // rcx
  int v76; // r8d
  int v77; // r9d
  __int64 v78; // rax
  _QWORD *v79; // rax
  __int64 v80; // rcx
  unsigned __int64 v81; // rdx
  __int64 v82; // rdx
  int v83; // eax
  int v84; // r9d
  __int64 v85; // rbx
  __int64 v86; // rax
  int v87; // eax
  int v88; // r8d
  int v89; // eax
  int v90; // r9d
  int v91; // eax
  int v92; // r8d
  const void *v93; // [rsp+0h] [rbp-260h]
  __int64 v96; // [rsp+18h] [rbp-248h]
  unsigned int v98; // [rsp+28h] [rbp-238h]
  unsigned __int8 v99; // [rsp+2Fh] [rbp-231h]
  __int64 v100; // [rsp+30h] [rbp-230h]
  __int64 v101; // [rsp+38h] [rbp-228h]
  __int64 v102; // [rsp+40h] [rbp-220h]
  __int64 v103; // [rsp+48h] [rbp-218h]
  __int64 v105; // [rsp+58h] [rbp-208h]
  _QWORD v106[64]; // [rsp+60h] [rbp-200h] BYREF

  v17 = *(_QWORD *)(a2 + 544);
  v18 = *(_QWORD *)(v17 - 48);
  v101 = v17;
  v103 = 0;
  if ( a15 )
    v103 = *(_QWORD *)(v18 - 24);
  v98 = -1;
  v99 = *(_BYTE *)(v18 + 16);
  v100 = v103 + 8;
  v96 = *(_QWORD *)(v17 + 40);
  *(_DWORD *)(a2 + 280) = 0;
  v19 = a1[1];
  v93 = (const void *)(a2 + 288);
  if ( !v19 )
  {
LABEL_20:
    v37 = *(_QWORD *)(a2 + 576) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)(a2 + 576) & 4) != 0 )
    {
      v38 = *(_QWORD ***)v37;
      v105 = *(_QWORD *)v37 + 8LL * *(unsigned int *)(v37 + 8);
    }
    else
    {
      v38 = (_QWORD **)(a2 + 576);
      if ( !v37 )
      {
LABEL_27:
        sub_15F20C0(*(_QWORD **)(a2 + 544));
        v47 = *(_DWORD *)(a3 + 24);
        if ( v47 )
        {
          v48 = v47 - 1;
          v49 = *(_QWORD *)(a2 + 544);
          v50 = *(_QWORD *)(a3 + 8);
          v51 = (v47 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
          v52 = (__int64 *)(v50 + 16LL * v51);
          v53 = *v52;
          if ( v49 == *v52 )
          {
LABEL_29:
            *v52 = -16;
            --*(_DWORD *)(a3 + 16);
            ++*(_DWORD *)(a3 + 20);
          }
          else
          {
            v89 = 1;
            while ( v53 != -8 )
            {
              v90 = v89 + 1;
              v51 = v48 & (v89 + v51);
              v52 = (__int64 *)(v50 + 16LL * v51);
              v53 = *v52;
              if ( v49 == *v52 )
                goto LABEL_29;
              v89 = v90;
            }
          }
        }
        sub_15F20C0(a1);
        v54 = *(_DWORD *)(a3 + 24);
        result = 1;
        if ( v54 )
        {
          v55 = v54 - 1;
          v56 = *(_QWORD *)(a3 + 8);
          v57 = (v54 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v58 = (_QWORD *)(v56 + 16LL * v57);
          v59 = (_QWORD *)*v58;
          if ( a1 == (_QWORD *)*v58 )
          {
LABEL_32:
            *v58 = -16;
            result = 1;
            --*(_DWORD *)(a3 + 16);
            ++*(_DWORD *)(a3 + 20);
          }
          else
          {
            v91 = 1;
            while ( v59 != (_QWORD *)-8LL )
            {
              v92 = v91 + 1;
              v57 = v55 & (v91 + v57);
              v58 = (_QWORD *)(v56 + 16LL * v57);
              v59 = (_QWORD *)*v58;
              if ( a1 == (_QWORD *)*v58 )
                goto LABEL_32;
              v91 = v92;
            }
            return 1;
          }
        }
        return result;
      }
      v105 = a2 + 584;
    }
    while ( (_QWORD **)v105 != v38 )
    {
      v39 = *v38;
      v40 = (__int64 *)sub_15F2050((__int64)a1);
      sub_15A5590((__int64)v106, v40, 0, 0);
      v41 = *(_QWORD *)(a2 + 544);
      sub_1AE9B50((__int64)v39, v41, v106);
      sub_15F20C0(v39);
      v42 = *(_DWORD *)(a3 + 24);
      if ( v42 )
      {
        v41 = (unsigned int)(v42 - 1);
        v43 = *(_QWORD *)(a3 + 8);
        v44 = v41 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v45 = (_QWORD *)(v43 + 16LL * v44);
        v46 = (_QWORD *)*v45;
        if ( v39 == (_QWORD *)*v45 )
        {
LABEL_25:
          *v45 = -16;
          --*(_DWORD *)(a3 + 16);
          ++*(_DWORD *)(a3 + 20);
        }
        else
        {
          v83 = 1;
          while ( v46 != (_QWORD *)-8LL )
          {
            v84 = v83 + 1;
            v44 = v41 & (v83 + v44);
            v45 = (_QWORD *)(v43 + 16LL * v44);
            v46 = (_QWORD *)*v45;
            if ( v39 == (_QWORD *)*v45 )
              goto LABEL_25;
            v83 = v84;
          }
        }
      }
      ++v38;
      sub_129E320((__int64)v106, v41);
    }
    goto LABEL_27;
  }
  v102 = a3;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v20 = v19;
        v19 = *(_QWORD *)(v19 + 8);
        v21 = sub_1648700(v20);
        v24 = *((_BYTE *)v21 + 16);
        if ( a15 )
          break;
LABEL_35:
        if ( v24 == 54 )
        {
          if ( v99 <= 0x17u )
            goto LABEL_39;
          v60 = v21[5];
          if ( v96 == v60 )
          {
            if ( v98 == -1 )
              v98 = sub_1B34670(v102, v101);
            if ( (unsigned int)sub_1B34670(v102, (__int64)v21) >= v98 )
            {
LABEL_39:
              if ( a15 )
                goto LABEL_53;
              v63 = *(_QWORD *)(v101 - 48);
              if ( v63 && v21 == (_QWORD *)v63 )
                v63 = sub_1599EF0((__int64 **)*v21);
              if ( a6
                && (v21[6] || *((__int16 *)v21 + 9) < 0)
                && sub_1625790((__int64)v21, 11)
                && !(unsigned __int8)sub_14BFF20(v63, a4, 0, a6, (__int64)v21, a5) )
              {
                sub_1B31B30(a6, (__int64 ***)v21);
              }
              sub_164D160((__int64)v21, v63, a7, a8, a9, a10, v22, v23, a13, a14);
              sub_15F20C0(v21);
              v64 = *(_DWORD *)(v102 + 24);
              if ( !v64 )
                goto LABEL_18;
              v65 = *(_QWORD *)(v102 + 8);
              v66 = v64 - 1;
              v67 = (v64 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
              v68 = (_QWORD *)(v65 + 16LL * v67);
              v69 = (_QWORD *)*v68;
              if ( v21 != (_QWORD *)*v68 )
              {
                v87 = 1;
                while ( v69 != (_QWORD *)-8LL )
                {
                  v88 = v87 + 1;
                  v67 = v66 & (v87 + v67);
                  v68 = (_QWORD *)(v65 + 16LL * v67);
                  v69 = (_QWORD *)*v68;
                  if ( v21 == (_QWORD *)*v68 )
                    goto LABEL_50;
                  v87 = v88;
                }
                goto LABEL_18;
              }
LABEL_50:
              *v68 = -16;
              --*(_DWORD *)(v102 + 16);
              ++*(_DWORD *)(v102 + 20);
              if ( !v19 )
                goto LABEL_19;
            }
            else
            {
              v78 = *(unsigned int *)(a2 + 280);
              if ( (unsigned int)v78 >= *(_DWORD *)(a2 + 284) )
              {
                sub_16CD150(a2 + 272, v93, 0, 8, v76, v77);
                v78 = *(unsigned int *)(a2 + 280);
              }
              *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8 * v78) = v96;
              ++*(_DWORD *)(a2 + 280);
              if ( !v19 )
                goto LABEL_19;
            }
          }
          else
          {
            if ( sub_15CC8F0(a5, v96, v60) )
              goto LABEL_39;
            v85 = v21[5];
            v86 = *(unsigned int *)(a2 + 280);
            if ( (unsigned int)v86 >= *(_DWORD *)(a2 + 284) )
            {
              sub_16CD150(a2 + 272, v93, 0, 8, v61, v62);
              v86 = *(unsigned int *)(a2 + 280);
            }
            *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8 * v86) = v85;
            ++*(_DWORD *)(a2 + 280);
            if ( !v19 )
              goto LABEL_19;
          }
        }
        else
        {
LABEL_18:
          if ( !v19 )
            goto LABEL_19;
        }
      }
      if ( v24 == 78 )
      {
        v25 = *((_DWORD *)v21 + 5) & 0xFFFFFFF;
        if ( *((char *)v21 + 23) < 0 )
        {
          v26 = sub_1648A40((__int64)v21);
          v28 = v26 + v27;
          if ( *((char *)v21 + 23) >= 0 )
          {
            if ( (unsigned int)(v28 >> 4) )
LABEL_109:
              BUG();
          }
          else if ( (unsigned int)((v28 - sub_1648A40((__int64)v21)) >> 4) )
          {
            if ( *((char *)v21 + 23) >= 0 )
              goto LABEL_109;
            v29 = *(_DWORD *)(sub_1648A40((__int64)v21) + 8);
            if ( *((char *)v21 + 23) >= 0 )
              BUG();
            v30 = sub_1648A40((__int64)v21);
            v32 = *(_DWORD *)(v30 + v31 - 4) - v29;
            goto LABEL_13;
          }
        }
        v32 = 0;
LABEL_13:
        v33 = v25 - 1 - v32;
        if ( v33 )
        {
          for ( i = 0; i != v33; ++i )
          {
            v35 = &v21[3 * (i - (*((_DWORD *)v21 + 5) & 0xFFFFFFF))];
            if ( a1 == (_QWORD *)*v35 && *v35 )
            {
              v73 = v35[1];
              v74 = v35[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v74 = v73;
              if ( v73 )
                *(_QWORD *)(v73 + 16) = *(_QWORD *)(v73 + 16) & 3LL | v74;
              *v35 = v103;
              if ( v103 )
              {
                v75 = *(_QWORD *)(v103 + 8);
                v35[1] = v75;
                if ( v75 )
                  *(_QWORD *)(v75 + 16) = (unsigned __int64)(v35 + 1) | *(_QWORD *)(v75 + 16) & 3LL;
                v35[2] = v100 | v35[2] & 3LL;
                *(_QWORD *)(v103 + 8) = v35;
              }
            }
          }
        }
        goto LABEL_18;
      }
      if ( v24 != 71 )
        break;
LABEL_53:
      if ( *(v21 - 3) )
      {
        v70 = *(v21 - 2);
        v71 = *(v21 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v71 = v70;
        if ( v70 )
          *(_QWORD *)(v70 + 16) = *(_QWORD *)(v70 + 16) & 3LL | v71;
      }
      *(v21 - 3) = v103;
      if ( !v103 )
        goto LABEL_18;
      v72 = *(_QWORD *)(v103 + 8);
      *(v21 - 2) = v72;
      if ( v72 )
        *(_QWORD *)(v72 + 16) = (unsigned __int64)(v21 - 2) | *(_QWORD *)(v72 + 16) & 3LL;
      *(v21 - 1) = v100 | *(v21 - 1) & 3LL;
      *(_QWORD *)(v103 + 8) = v21 - 3;
      if ( !v19 )
        goto LABEL_19;
    }
    if ( v24 != 56 )
      goto LABEL_35;
    v79 = &v21[-3 * (*((_DWORD *)v21 + 5) & 0xFFFFFFF)];
    if ( *v79 )
    {
      v80 = v79[1];
      v81 = v79[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v81 = v80;
      if ( v80 )
        *(_QWORD *)(v80 + 16) = *(_QWORD *)(v80 + 16) & 3LL | v81;
    }
    *v79 = v103;
    if ( !v103 )
      goto LABEL_18;
    v82 = *(_QWORD *)(v103 + 8);
    v79[1] = v82;
    if ( v82 )
      *(_QWORD *)(v82 + 16) = (unsigned __int64)(v79 + 1) | *(_QWORD *)(v82 + 16) & 3LL;
    v79[2] = v100 | v79[2] & 3LL;
    *(_QWORD *)(v103 + 8) = v79;
  }
  while ( v19 );
LABEL_19:
  a3 = v102;
  result = 0;
  if ( !*(_DWORD *)(a2 + 280) )
    goto LABEL_20;
  return result;
}
