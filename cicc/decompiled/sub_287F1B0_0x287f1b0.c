// Function: sub_287F1B0
// Address: 0x287f1b0
//
__int64 __fastcall sub_287F1B0(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 result; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // r10
  int v21; // edi
  int v22; // esi
  int v23; // r11d
  _QWORD *v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // rbx
  _BYTE *v27; // r12
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdi
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rcx
  unsigned int v36; // edx
  __int64 v37; // r12
  const void *v38; // r13
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r12
  __int64 *v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rdx
  _QWORD *v49; // rax
  _QWORD *v50; // rdx
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // r8
  unsigned __int8 **v54; // r14
  __int64 v55; // rax
  unsigned __int8 **v56; // r11
  __int64 v57; // rcx
  unsigned __int64 v58; // rdi
  _QWORD *v59; // r15
  __int64 v60; // rax
  unsigned __int8 **v61; // r13
  __int64 v62; // r11
  __int64 v63; // r12
  __int64 v64; // r8
  int v65; // esi
  unsigned __int8 *v66; // rbx
  __int64 v67; // r9
  int v68; // esi
  unsigned int v69; // ecx
  unsigned __int8 **v70; // rdx
  unsigned __int8 *v71; // r10
  unsigned __int8 *v72; // rdx
  unsigned __int8 **v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rdx
  unsigned __int64 v78; // rax
  int v79; // edx
  __int64 v80; // rax
  bool v81; // cc
  __int64 v82; // [rsp+8h] [rbp-88h]
  __int64 v83; // [rsp+10h] [rbp-80h]
  int v84; // [rsp+10h] [rbp-80h]
  __int64 v85; // [rsp+20h] [rbp-70h]
  int v86; // [rsp+28h] [rbp-68h]
  unsigned __int8 **v88; // [rsp+30h] [rbp-60h] BYREF
  __int64 v89; // [rsp+38h] [rbp-58h]
  _BYTE v90[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = *a1;
  v7 = *(unsigned int *)(*a1 + 8);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
  {
    sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(v6 + 8);
  }
  v8 = (__int64)a1;
  *(_QWORD *)(*(_QWORD *)v6 + 8 * v7) = a2;
  ++*(_DWORD *)(v6 + 8);
  v9 = sub_B43CB0(a2);
  v86 = ((unsigned __int8)sub_B2D610(v9, 18) == 0) + 2;
  while ( 2 )
  {
    v12 = *(_QWORD *)v8;
    LODWORD(result) = *(_DWORD *)(*(_QWORD *)v8 + 8LL);
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(*(_QWORD *)v12 + 8LL * (unsigned int)result - 8);
          *(_DWORD *)(v12 + 8) = result - 1;
          v15 = *(_QWORD *)(v8 + 8);
          v16 = *(unsigned int *)(v15 + 24);
          v17 = *(_QWORD *)(v15 + 8);
          if ( (_DWORD)v16 )
          {
            v11 = (unsigned int)(v16 - 1);
            v85 = (unsigned int)(((4 * a3) >> 2) + 4 * (((4 * a3) >> 2) + 8 * a3));
            v18 = v11
                & (((0xBF58476D1CE4E5B9LL
                   * (v85 | ((unsigned __int64)(((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v85));
            v19 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v19;
            v21 = (4 * *((_DWORD *)v19 + 2)) >> 2;
            v10 = (unsigned int)((4 * a3) >> 2);
            if ( v21 != (_DWORD)v10 || v14 != v20 )
            {
              v22 = 1;
              while ( v21 || v20 != -4096 )
              {
                v23 = v22 + 1;
                v18 = v11 & (v18 + v22);
                v19 = (__int64 *)(v17 + 16LL * v18);
                v20 = *v19;
                v21 = (4 * *((_DWORD *)v19 + 2)) >> 2;
                if ( v14 == *v19 && v21 == (_DWORD)v10 )
                  goto LABEL_33;
                v22 = v23;
              }
              goto LABEL_28;
            }
LABEL_33:
            if ( v19 != (__int64 *)(16 * v16 + v17) && *((char *)v19 + 11) >= 0 )
              break;
          }
LABEL_28:
          v12 = *(_QWORD *)v8;
          result = *(unsigned int *)(*(_QWORD *)v8 + 8LL);
          if ( !(_DWORD)result )
            goto LABEL_29;
        }
        *((_BYTE *)v19 + 11) |= 0x80u;
        if ( *(_BYTE *)v14 != 84 || (v39 = **(_QWORD **)(v8 + 16), **(_QWORD **)(v39 + 32) != *(_QWORD *)(v14 + 40)) )
        {
          v24 = (_QWORD *)v14;
          if ( (*((_BYTE *)v19 + 11) & 0x40) != 0 )
            goto LABEL_15;
          v53 = *(_QWORD *)(v8 + 32);
          v88 = (unsigned __int8 **)v90;
          v89 = 0x400000000LL;
          if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
          {
            v54 = *(unsigned __int8 ***)(v14 - 8);
            v55 = 4LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
            v56 = &v54[v55];
            if ( v54 != &v54[v55] )
              goto LABEL_57;
          }
          else
          {
            v56 = (unsigned __int8 **)v14;
            v80 = 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
            v54 = (unsigned __int8 **)(v14 - v80);
            if ( v14 - v80 != v14 )
            {
LABEL_57:
              v57 = v8;
              v58 = 4;
              v59 = (_QWORD *)v14;
              v60 = 0;
              v61 = v56;
              v62 = v14;
              v63 = v53;
              v64 = v57;
              while ( 1 )
              {
                v65 = *(_DWORD *)(v63 + 24);
                v66 = *v54;
                v67 = *(_QWORD *)(v63 + 8);
                if ( v65 )
                {
                  v68 = v65 - 1;
                  v69 = v68 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
                  v70 = (unsigned __int8 **)(v67 + 16LL * v69);
                  v71 = *v70;
                  if ( v66 == *v70 )
                  {
LABEL_61:
                    v72 = v70[1];
                    if ( v72 )
                      v66 = v72;
                  }
                  else
                  {
                    v79 = 1;
                    while ( v71 != (unsigned __int8 *)-4096LL )
                    {
                      v69 = v68 & (v79 + v69);
                      v84 = v79 + 1;
                      v70 = (unsigned __int8 **)(v67 + 16LL * v69);
                      v71 = *v70;
                      if ( v66 == *v70 )
                        goto LABEL_61;
                      v79 = v84;
                    }
                  }
                }
                if ( v60 + 1 > v58 )
                {
                  v82 = v64;
                  v83 = v62;
                  sub_C8D5F0((__int64)&v88, v90, v60 + 1, 8u, v64, v67);
                  v60 = (unsigned int)v89;
                  v64 = v82;
                  v62 = v83;
                }
                v54 += 4;
                v88[v60] = v66;
                v60 = (unsigned int)(v89 + 1);
                LODWORD(v89) = v89 + 1;
                if ( v54 == v61 )
                  break;
                v58 = HIDWORD(v89);
              }
              v73 = v88;
              v24 = v59;
              v14 = v62;
              v8 = v64;
              v74 = (unsigned int)v60;
              goto LABEL_67;
            }
          }
          v73 = (unsigned __int8 **)v90;
          v74 = 0;
LABEL_67:
          v75 = sub_DFCEF0(*(__int64 ***)(v8 + 48), (unsigned __int8 *)v14, v73, v74, v86);
          v10 = v76;
          v77 = *(_QWORD *)(v8 + 40);
          if ( (_DWORD)v10 == 1 )
            *(_DWORD *)(v77 + 8) = 1;
          if ( __OFADD__(*(_QWORD *)v77, v75) )
          {
            v81 = v75 <= 0;
            v78 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v81 )
              v78 = 0x8000000000000000LL;
          }
          else
          {
            v78 = *(_QWORD *)v77 + v75;
          }
          *(_QWORD *)v77 = v78;
          if ( v88 != (unsigned __int8 **)v90 )
            _libc_free((unsigned __int64)v88);
LABEL_15:
          v25 = 4LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
          v26 = (_QWORD *)(v14 - v25 * 8);
          if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
          {
            v26 = *(_QWORD **)(v14 - 8);
            v24 = &v26[v25];
          }
          if ( v26 == v24 )
            goto LABEL_28;
          while ( 1 )
          {
            v27 = (_BYTE *)*v26;
            if ( *(_BYTE *)*v26 > 0x1Cu )
            {
              v28 = *(__int64 **)(v8 + 16);
              v29 = *((_QWORD *)v27 + 5);
              v30 = *v28;
              if ( *(_BYTE *)(*v28 + 84) )
              {
                v31 = *(_QWORD **)(v30 + 64);
                v32 = &v31[*(unsigned int *)(v30 + 76)];
                if ( v31 != v32 )
                {
                  while ( v29 != *v31 )
                  {
                    if ( v32 == ++v31 )
                      goto LABEL_27;
                  }
LABEL_24:
                  v33 = *(_QWORD *)v8;
                  v34 = *(unsigned int *)(*(_QWORD *)v8 + 8LL);
                  if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(*(_QWORD *)v8 + 12LL) )
                  {
                    sub_C8D5F0(*(_QWORD *)v8, (const void *)(v33 + 16), v34 + 1, 8u, v10, v11);
                    v34 = *(unsigned int *)(v33 + 8);
                  }
                  *(_QWORD *)(*(_QWORD *)v33 + 8 * v34) = v27;
                  ++*(_DWORD *)(v33 + 8);
                }
              }
              else if ( sub_C8CA60(v30 + 56, v29) )
              {
                goto LABEL_24;
              }
            }
LABEL_27:
            v26 += 4;
            if ( v24 == v26 )
              goto LABEL_28;
          }
        }
        if ( !a3 )
          goto LABEL_28;
        v40 = sub_D47930(v39);
        v41 = *(_QWORD *)(v14 - 8);
        v42 = v40;
        if ( (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) != 0 )
        {
          v43 = 0;
          while ( v42 != *(_QWORD *)(v41 + 32LL * *(unsigned int *)(v14 + 72) + 8 * v43) )
          {
            if ( (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) == (_DWORD)++v43 )
              goto LABEL_79;
          }
          v44 = 32 * v43;
        }
        else
        {
LABEL_79:
          v44 = 0x1FFFFFFFE0LL;
        }
        v45 = *(_QWORD *)(v41 + v44);
        if ( *(_BYTE *)v45 <= 0x1Cu )
          goto LABEL_28;
        v46 = *(__int64 **)(v8 + 16);
        v47 = *(_QWORD *)(v45 + 40);
        v48 = *v46;
        if ( *(_BYTE *)(*v46 + 84) )
          break;
        if ( sub_C8CA60(v48 + 56, v47) )
          goto LABEL_49;
        v12 = *(_QWORD *)v8;
        result = *(unsigned int *)(*(_QWORD *)v8 + 8LL);
        if ( !(_DWORD)result )
          goto LABEL_29;
      }
      v49 = *(_QWORD **)(v48 + 64);
      v50 = &v49[*(unsigned int *)(v48 + 76)];
      if ( v49 == v50 )
        goto LABEL_28;
      while ( v47 != *v49 )
      {
        if ( v50 == ++v49 )
          goto LABEL_28;
      }
LABEL_49:
      v51 = *(_QWORD *)(v8 + 24);
      v52 = *(unsigned int *)(v51 + 8);
      if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(v51 + 12) )
      {
        sub_C8D5F0(*(_QWORD *)(v8 + 24), (const void *)(v51 + 16), v52 + 1, 8u, v10, v11);
        v52 = *(unsigned int *)(v51 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v51 + 8 * v52) = v45;
      ++*(_DWORD *)(v51 + 8);
      v12 = *(_QWORD *)v8;
      result = *(unsigned int *)(*(_QWORD *)v8 + 8LL);
    }
    while ( (_DWORD)result );
LABEL_29:
    v35 = *(_QWORD *)(v8 + 24);
    v36 = *(_DWORD *)(v35 + 8);
    if ( v36 )
    {
      v37 = v36;
      v38 = *(const void **)v35;
      if ( v36 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
      {
        sub_C8D5F0(v12, (const void *)(v12 + 16), v36, 8u, v10, v11);
        result = *(unsigned int *)(v12 + 8);
      }
      memcpy((void *)(*(_QWORD *)v12 + 8 * result), v38, 8 * v37);
      *(_DWORD *)(v12 + 8) += v37;
      --a3;
      *(_DWORD *)(*(_QWORD *)(v8 + 24) + 8LL) = 0;
      continue;
    }
    return result;
  }
}
