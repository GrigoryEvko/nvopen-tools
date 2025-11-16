// Function: sub_29B45E0
// Address: 0x29b45e0
//
__int64 *__fastcall sub_29B45E0(unsigned int **a1)
{
  __int64 *result; // rax
  __int64 v2; // r15
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r9
  unsigned int *v8; // rdx
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rcx
  unsigned __int64 v12; // r8
  void *v13; // r11
  size_t v14; // r10
  __int64 v15; // rax
  _DWORD *v16; // r8
  _QWORD *v17; // rax
  __int64 v18; // r9
  unsigned int *v19; // rdx
  __int64 v20; // r8
  _QWORD *v21; // r15
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned int v25; // ecx
  __int64 v26; // rcx
  unsigned __int64 *v27; // rcx
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 *v31; // rdi
  __int64 *v32; // rbx
  _QWORD *v33; // r14
  __int64 v34; // r13
  __int64 v35; // r15
  char *v36; // rdi
  __int64 v37; // rdx
  char *v38; // rsi
  __int64 v39; // r8
  char *v40; // rax
  char *v41; // r8
  __int64 v42; // rax
  const char *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 *v47; // r12
  __int64 *v48; // r14
  __int64 v49; // r15
  char *v50; // rax
  __int64 v51; // rsi
  unsigned int v52; // eax
  unsigned __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // r10
  _QWORD *v56; // rax
  __int64 *v57; // rdi
  _QWORD *v58; // rax
  char *v59; // rax
  __int64 v60; // r13
  char *v61; // rbx
  unsigned int *v62; // r14
  unsigned __int64 v63; // r8
  unsigned __int64 v64; // rdi
  __int64 *v65; // [rsp+10h] [rbp-2D0h]
  unsigned __int64 v66; // [rsp+18h] [rbp-2C8h]
  __int64 v67; // [rsp+20h] [rbp-2C0h]
  size_t nb; // [rsp+28h] [rbp-2B8h]
  size_t n; // [rsp+28h] [rbp-2B8h]
  size_t na; // [rsp+28h] [rbp-2B8h]
  _DWORD *src; // [rsp+30h] [rbp-2B0h]
  _QWORD *srca; // [rsp+30h] [rbp-2B0h]
  void *srcd; // [rsp+30h] [rbp-2B0h]
  char *srcb; // [rsp+30h] [rbp-2B0h]
  unsigned int *srcc; // [rsp+30h] [rbp-2B0h]
  unsigned int *v76; // [rsp+38h] [rbp-2A8h]
  __int64 v77; // [rsp+38h] [rbp-2A8h]
  __int64 v78; // [rsp+38h] [rbp-2A8h]
  __int64 *v79; // [rsp+38h] [rbp-2A8h]
  unsigned int v80; // [rsp+38h] [rbp-2A8h]
  int v81; // [rsp+38h] [rbp-2A8h]
  int v82; // [rsp+38h] [rbp-2A8h]
  unsigned int *v83; // [rsp+38h] [rbp-2A8h]
  __int64 v84; // [rsp+38h] [rbp-2A8h]
  __int64 *v86; // [rsp+48h] [rbp-298h]
  __int64 *v87; // [rsp+50h] [rbp-290h] BYREF
  __int64 v88; // [rsp+58h] [rbp-288h]
  _BYTE v89[64]; // [rsp+60h] [rbp-280h] BYREF
  const char *v90; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v91; // [rsp+A8h] [rbp-238h]
  _QWORD v92[2]; // [rsp+B0h] [rbp-230h] BYREF
  __int16 v93; // [rsp+C0h] [rbp-220h]

  result = (__int64 *)a1[11];
  v86 = result;
  v65 = &result[*((unsigned int *)a1 + 24)];
  if ( result != v65 )
  {
    do
    {
      v2 = *v86;
      v3 = (__int64 *)(*(_QWORD *)(*v86 + 48) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v3 == (__int64 *)(*v86 + 48) )
        goto LABEL_106;
      if ( !v3 )
        BUG();
      if ( (unsigned int)*((unsigned __int8 *)v3 - 24) - 30 > 0xA )
LABEL_106:
        BUG();
      if ( *((_BYTE *)v3 - 24) != 30 )
        goto LABEL_3;
      v90 = sub_BD5D20(*v86);
      v92[0] = ".ret";
      v4 = v67;
      v93 = 773;
      LOWORD(v4) = 0;
      v91 = v5;
      v67 = v4;
      v6 = sub_AA8550((_QWORD *)v2, v3, v4, (__int64)&v90, 0);
      v8 = *a1;
      if ( !*a1 )
        goto LABEL_3;
      v9 = *(_DWORD *)(v2 + 44);
      v10 = (unsigned int)(v9 + 1);
      if ( (unsigned int)v10 >= v8[8] )
        BUG();
      v11 = *(_QWORD *)(*((_QWORD *)v8 + 3) + 8 * v10);
      v12 = *(unsigned int *)(v11 + 32);
      v13 = *(void **)(v11 + 24);
      v87 = (__int64 *)v89;
      v88 = 0x800000000LL;
      v14 = 8 * v12;
      if ( v12 > 8 )
      {
        nb = 8 * v12;
        srcd = v13;
        v81 = v12;
        sub_C8D5F0((__int64)&v87, v89, v12, 8u, v12, v7);
        LODWORD(v12) = v81;
        v13 = srcd;
        v14 = nb;
        v57 = &v87[(unsigned int)v88];
      }
      else
      {
        if ( !v14 )
          goto LABEL_12;
        v57 = (__int64 *)v89;
      }
      v82 = v12;
      memcpy(v57, v13, v14);
      LODWORD(v14) = v88;
      LODWORD(v12) = v82;
      v8 = *a1;
      v9 = *(_DWORD *)(v2 + 44);
LABEL_12:
      v15 = (unsigned int)(v9 + 1);
      LODWORD(v88) = v14 + v12;
      if ( (unsigned int)v15 < v8[8] )
      {
        v76 = v8;
        v16 = *(_DWORD **)(*((_QWORD *)v8 + 3) + 8 * v15);
        *((_BYTE *)v8 + 112) = 0;
        src = v16;
        v17 = (_QWORD *)sub_22077B0(0x50u);
        v19 = v76;
        v20 = (__int64)src;
        v21 = v17;
        if ( !v17 )
          goto LABEL_17;
        *v17 = v6;
        v17[1] = src;
        if ( src )
          v22 = src[4] + 1;
        else
          v22 = 0;
        goto LABEL_16;
      }
      *((_BYTE *)v8 + 112) = 0;
      v83 = v8;
      v58 = (_QWORD *)sub_22077B0(0x50u);
      v19 = v83;
      v21 = v58;
      if ( v58 )
      {
        *v58 = v6;
        v20 = 0;
        v22 = 0;
        v21[1] = 0;
LABEL_16:
        *((_DWORD *)v21 + 4) = v22;
        v21[3] = v21 + 5;
        v21[4] = 0x400000000LL;
        v21[9] = -1;
        goto LABEL_17;
      }
      v20 = 0;
LABEL_17:
      if ( v6 )
      {
        v23 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
        v24 = 8 * v23;
      }
      else
      {
        v24 = 0;
        LODWORD(v23) = 0;
      }
      v25 = v19[8];
      if ( v25 > (unsigned int)v23 )
        goto LABEL_20;
      v51 = *((_QWORD *)v19 + 13);
      v52 = v23 + 1;
      if ( *(_DWORD *)(v51 + 88) >= v52 )
        v52 = *(_DWORD *)(v51 + 88);
      v80 = v52;
      v53 = v52;
      v54 = v25;
      if ( v53 == v25 )
      {
LABEL_20:
        v26 = *((_QWORD *)v19 + 3);
      }
      else
      {
        v55 = 8 * v53;
        if ( v53 < v25 )
        {
          v26 = *((_QWORD *)v19 + 3);
          v59 = (char *)(v26 + 8 * v54);
          srcb = (char *)(v26 + v55);
          if ( v59 != (char *)(v26 + v55) )
          {
            n = v24;
            v60 = v20;
            v61 = v59;
            v62 = v19;
            do
            {
              v63 = *((_QWORD *)v61 - 1);
              v61 -= 8;
              if ( v63 )
              {
                v64 = *(_QWORD *)(v63 + 24);
                if ( v64 != v63 + 40 )
                {
                  v66 = v63;
                  _libc_free(v64);
                  v63 = v66;
                }
                j_j___libc_free_0(v63);
              }
            }
            while ( srcb != v61 );
            v24 = n;
            v26 = *((_QWORD *)v62 + 3);
            v20 = v60;
            v19 = v62;
          }
        }
        else
        {
          if ( v53 > v19[9] )
          {
            na = v20;
            srcc = v19;
            sub_B1B4E0((__int64)(v19 + 6), v53);
            v19 = srcc;
            v55 = 8 * v53;
            v20 = na;
            v54 = srcc[8];
          }
          v26 = *((_QWORD *)v19 + 3);
          v56 = (_QWORD *)(v26 + 8 * v54);
          if ( v56 != (_QWORD *)(v26 + v55) )
          {
            do
            {
              if ( v56 )
                *v56 = 0;
              ++v56;
            }
            while ( (_QWORD *)(v26 + v55) != v56 );
            v26 = *((_QWORD *)v19 + 3);
          }
        }
        v19[8] = v80;
      }
      v27 = (unsigned __int64 *)(v24 + v26);
      v28 = *v27;
      *v27 = (unsigned __int64)v21;
      if ( v28 )
      {
        v29 = *(_QWORD *)(v28 + 24);
        if ( v29 != v28 + 40 )
        {
          v77 = v20;
          _libc_free(v29);
          v20 = v77;
        }
        v78 = v20;
        j_j___libc_free_0(v28);
        v20 = v78;
      }
      if ( v20 )
      {
        v30 = *(unsigned int *)(v20 + 32);
        if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 36) )
        {
          v84 = v20;
          sub_C8D5F0(v20 + 24, (const void *)(v20 + 40), v30 + 1, 8u, v20, v18);
          v20 = v84;
          v30 = *(unsigned int *)(v84 + 32);
        }
        *(_QWORD *)(*(_QWORD *)(v20 + 24) + 8 * v30) = v21;
        ++*(_DWORD *)(v20 + 32);
      }
      v31 = v87;
      v79 = &v87[(unsigned int)v88];
      if ( v79 != v87 )
      {
        v32 = v87;
        v33 = v21;
        while ( 1 )
        {
          v34 = *v32;
          *((_BYTE *)*a1 + 112) = 0;
          v35 = *(_QWORD *)(v34 + 8);
          if ( v33 != (_QWORD *)v35 )
            break;
LABEL_56:
          if ( v79 == ++v32 )
          {
            v31 = v87;
            goto LABEL_58;
          }
        }
        v36 = *(char **)(v35 + 24);
        v37 = *(unsigned int *)(v35 + 32);
        v38 = &v36[8 * v37];
        v39 = (8 * v37) >> 3;
        if ( (8 * v37) >> 5 )
        {
          v40 = &v36[32 * ((8 * v37) >> 5)];
          while ( v34 != *(_QWORD *)v36 )
          {
            if ( v34 == *((_QWORD *)v36 + 1) )
            {
              v36 += 8;
              v41 = v36 + 8;
              goto LABEL_40;
            }
            if ( v34 == *((_QWORD *)v36 + 2) )
            {
              v36 += 16;
              v41 = v36 + 8;
              goto LABEL_40;
            }
            if ( v34 == *((_QWORD *)v36 + 3) )
            {
              v36 += 24;
              v41 = v36 + 8;
              goto LABEL_40;
            }
            v36 += 32;
            if ( v40 == v36 )
            {
              v39 = (v38 - v36) >> 3;
              goto LABEL_61;
            }
          }
LABEL_39:
          v41 = v36 + 8;
LABEL_40:
          if ( v41 != v38 )
          {
            memmove(v36, v41, v38 - v41);
            LODWORD(v37) = *(_DWORD *)(v35 + 32);
          }
          *(_DWORD *)(v35 + 32) = v37 - 1;
          *(_QWORD *)(v34 + 8) = v33;
          v42 = *((unsigned int *)v33 + 8);
          if ( v42 + 1 > (unsigned __int64)*((unsigned int *)v33 + 9) )
          {
            sub_C8D5F0((__int64)(v33 + 3), v33 + 5, v42 + 1, 8u, (__int64)v41, v18);
            v42 = *((unsigned int *)v33 + 8);
          }
          *(_QWORD *)(v33[3] + 8 * v42) = v34;
          ++*((_DWORD *)v33 + 8);
          if ( *(_DWORD *)(v34 + 16) != *(_DWORD *)(*(_QWORD *)(v34 + 8) + 16LL) + 1 )
          {
            srca = v33;
            v90 = (const char *)v92;
            v43 = (const char *)v92;
            v92[0] = v34;
            v91 = 0x4000000001LL;
            LODWORD(v44) = 1;
            do
            {
              v45 = (unsigned int)v44;
              v44 = (unsigned int)(v44 - 1);
              v46 = *(_QWORD *)&v43[8 * v45 - 8];
              LODWORD(v91) = v44;
              v47 = *(__int64 **)(v46 + 24);
              *(_DWORD *)(v46 + 16) = *(_DWORD *)(*(_QWORD *)(v46 + 8) + 16LL) + 1;
              v48 = &v47[*(unsigned int *)(v46 + 32)];
              if ( v47 != v48 )
              {
                do
                {
                  v49 = *v47;
                  if ( *(_DWORD *)(*v47 + 16) != *(_DWORD *)(*(_QWORD *)(*v47 + 8) + 16LL) + 1 )
                  {
                    if ( v44 + 1 > (unsigned __int64)HIDWORD(v91) )
                    {
                      sub_C8D5F0((__int64)&v90, v92, v44 + 1, 8u, (__int64)v41, v18);
                      v44 = (unsigned int)v91;
                    }
                    *(_QWORD *)&v90[8 * v44] = v49;
                    v44 = (unsigned int)(v91 + 1);
                    LODWORD(v91) = v91 + 1;
                  }
                  ++v47;
                }
                while ( v48 != v47 );
                v43 = v90;
              }
            }
            while ( (_DWORD)v44 );
            v33 = srca;
            if ( v43 != (const char *)v92 )
              _libc_free((unsigned __int64)v43);
          }
          goto LABEL_56;
        }
LABEL_61:
        switch ( v39 )
        {
          case 2LL:
            v50 = v36;
            v36 += 8;
            if ( v34 != *(_QWORD *)v50 )
              goto LABEL_69;
            break;
          case 3LL:
            v41 = v36 + 8;
            v50 = v36 + 8;
            if ( v34 == *(_QWORD *)v36 )
              goto LABEL_40;
            v36 += 16;
            if ( v34 != *(_QWORD *)v50 )
              goto LABEL_69;
            break;
          case 1LL:
LABEL_69:
            if ( v34 == *(_QWORD *)v36 )
              goto LABEL_39;
            goto LABEL_64;
          default:
LABEL_64:
            v36 = v38;
            v41 = v38 + 8;
            goto LABEL_40;
        }
        v36 = v50;
        goto LABEL_39;
      }
LABEL_58:
      if ( v31 != (__int64 *)v89 )
        _libc_free((unsigned __int64)v31);
LABEL_3:
      result = ++v86;
    }
    while ( v65 != v86 );
  }
  return result;
}
