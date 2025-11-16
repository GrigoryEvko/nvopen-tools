// Function: sub_EDD200
// Address: 0xedd200
//
__int64 __fastcall sub_EDD200(_QWORD *a1, __int64 a2, __int64 a3, unsigned int *a4, unsigned __int64 a5)
{
  __int64 v5; // r13
  __int64 v7; // r15
  _QWORD *v8; // r13
  _QWORD *v9; // r14
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  _QWORD *v12; // rbx
  _QWORD *v13; // r15
  __int64 v14; // rdi
  unsigned int *v15; // rax
  char *v16; // rbx
  __int64 *v17; // r12
  unsigned __int64 *v18; // rdx
  __int64 v19; // rcx
  bool v20; // zf
  unsigned int *v21; // rax
  unsigned __int64 v22; // r13
  __int64 v23; // r14
  __int64 *v24; // rax
  unsigned __int64 v25; // r12
  __int64 v26; // r13
  _BYTE *v27; // rsi
  __int64 v28; // rdx
  __int64 *v29; // rsi
  _BYTE *v30; // r14
  _BYTE *v31; // r11
  _BYTE *v32; // rax
  __int64 v33; // rdi
  _BYTE *v34; // r10
  _BYTE *v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r9
  __int64 v38; // r8
  const void *v39; // rdi
  unsigned int *v40; // rax
  __int64 v41; // rcx
  __int64 i; // r14
  _BYTE *v43; // rsi
  __int64 v44; // rdx
  _QWORD *v45; // r12
  _QWORD *v46; // r14
  _QWORD *v47; // r13
  _QWORD *v48; // rdi
  _QWORD *v49; // rbx
  _QWORD *v50; // r15
  __int64 v51; // rdi
  __int64 v52; // rax
  _QWORD *v54; // [rsp+8h] [rbp-B8h]
  __int64 v56; // [rsp+10h] [rbp-B0h]
  signed __int64 v57; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  __int64 v59; // [rsp+18h] [rbp-A8h]
  __int64 v60; // [rsp+18h] [rbp-A8h]
  _QWORD *v61; // [rsp+20h] [rbp-A0h]
  _QWORD *v62; // [rsp+20h] [rbp-A0h]
  unsigned int *v63; // [rsp+28h] [rbp-98h] BYREF
  __int64 v64; // [rsp+30h] [rbp-90h] BYREF
  __int64 v65; // [rsp+38h] [rbp-88h]
  __int64 v66; // [rsp+40h] [rbp-80h] BYREF
  __int64 v67; // [rsp+48h] [rbp-78h] BYREF
  _BYTE *v68; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v69; // [rsp+58h] [rbp-68h]
  _BYTE *v70; // [rsp+60h] [rbp-60h]
  _BYTE *v71; // [rsp+70h] [rbp-50h] BYREF
  _BYTE *v72; // [rsp+78h] [rbp-48h]
  _BYTE *v73; // [rsp+80h] [rbp-40h]

  v64 = a2;
  v65 = a3;
  v63 = a4;
  v57 = a5 & 7;
  if ( (a5 & 7) != 0 )
    return 0;
  v7 = (__int64)a1;
  v54 = (_QWORD *)*a1;
  v61 = (_QWORD *)a1[1];
  if ( (_QWORD *)*a1 != v61 )
  {
    v8 = (_QWORD *)*a1;
    do
    {
      v9 = (_QWORD *)v8[6];
      if ( v9 )
      {
        v10 = v9 + 9;
        do
        {
          v11 = (_QWORD *)*(v10 - 3);
          v12 = (_QWORD *)*(v10 - 2);
          v10 -= 3;
          v13 = v11;
          if ( v12 != v11 )
          {
            do
            {
              if ( *v13 )
                j_j___libc_free_0(*v13, v13[2] - *v13);
              v13 += 3;
            }
            while ( v12 != v13 );
            v11 = (_QWORD *)*v10;
          }
          if ( v11 )
            j_j___libc_free_0(v11, v10[2] - (_QWORD)v11);
        }
        while ( v9 != v10 );
        j_j___libc_free_0(v9, 72);
      }
      v14 = v8[3];
      if ( v14 )
        j_j___libc_free_0(v14, v8[5] - v14);
      if ( *v8 )
        j_j___libc_free_0(*v8, v8[2] - *v8);
      v8 += 10;
    }
    while ( v61 != v8 );
    v7 = (__int64)a1;
    a1[1] = v54;
  }
  v15 = v63;
  v68 = 0;
  v69 = 0;
  v16 = (char *)v63 + a5;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  if ( v63 < (unsigned int *)((char *)v63 + a5) )
  {
    v17 = (__int64 *)&v68;
    while ( 1 )
    {
      v18 = (unsigned __int64 *)(v15 + 2);
      if ( v15 + 2 >= (unsigned int *)v16 )
        goto LABEL_81;
      v19 = *(_QWORD *)v15;
      v20 = *(_DWORD *)(v7 + 28) == 1;
      v63 = v15 + 2;
      v66 = v19;
      if ( v20 )
      {
        v22 = (a5 >> 3) - 1;
      }
      else
      {
        v21 = v15 + 4;
        if ( v21 > (unsigned int *)v16 )
          goto LABEL_81;
        v22 = *v18;
        v63 = v21;
        v18 = (unsigned __int64 *)v21;
      }
      if ( v16 < (char *)&v18[v22] )
      {
LABEL_81:
        v39 = v71;
        v5 = 0;
        v57 = v73 - v71;
        goto LABEL_82;
      }
      if ( v68 != v69 )
        v69 = v68;
      v23 = 0;
      sub_9C9810((__int64)v17, v22);
      if ( v22 )
      {
        v24 = v17;
        v25 = v22;
        v26 = (__int64)v24;
        do
        {
          while ( 1 )
          {
            v27 = v69;
            v28 = *(_QWORD *)v63;
            v63 += 2;
            v67 = v28;
            if ( v69 != v70 )
              break;
            ++v23;
            sub_A235E0(v26, v69, &v67);
            if ( v25 == v23 )
              goto LABEL_39;
          }
          if ( v69 )
          {
            *(_QWORD *)v69 = v28;
            v27 = v69;
          }
          ++v23;
          v69 = v27 + 8;
        }
        while ( v25 != v23 );
LABEL_39:
        v17 = (__int64 *)v26;
      }
      if ( *(_DWORD *)(v7 + 28) <= 0xAu )
        goto LABEL_41;
      v40 = v63 + 2;
      if ( v63 + 2 > (unsigned int *)v16 )
        goto LABEL_81;
      v41 = *(_QWORD *)v63;
      v63 += 2;
      if ( v16 < (char *)v40 + v41 )
        goto LABEL_81;
      if ( v72 != v71 )
        v72 = v71;
      v58 = v41;
      sub_ED8CE0((__int64)&v71, v41);
      if ( v58 )
      {
        for ( i = 0; i != v58; ++i )
        {
          while ( 1 )
          {
            v43 = v72;
            v44 = *(_QWORD *)v63;
            v63 += 2;
            LOBYTE(v67) = v44;
            if ( v72 != v73 )
              break;
            ++i;
            sub_C8FB10((__int64)&v71, v72, (char *)&v67);
            if ( i == v58 )
              goto LABEL_59;
          }
          if ( v72 )
          {
            *v72 = v44;
            v43 = v72;
          }
          v72 = v43 + 1;
        }
LABEL_59:
        v29 = *(__int64 **)(v7 + 8);
        if ( v29 == *(__int64 **)(v7 + 16) )
        {
LABEL_60:
          sub_ED9210((__int64 *)v7, v29, &v64, &v66, v17, (__int64 *)&v71);
          if ( *(_DWORD *)(v7 + 28) <= 2u )
            goto LABEL_45;
          goto LABEL_61;
        }
      }
      else
      {
LABEL_41:
        v29 = *(__int64 **)(v7 + 8);
        if ( v29 == *(__int64 **)(v7 + 16) )
          goto LABEL_60;
      }
      v30 = v68;
      v31 = v69;
      v68 = 0;
      v32 = v70;
      v33 = (__int64)v71;
      v70 = 0;
      v34 = v72;
      v35 = v73;
      v69 = 0;
      v73 = 0;
      v36 = v66;
      v72 = 0;
      v37 = v64;
      v71 = 0;
      v38 = v65;
      if ( v29 )
      {
        *v29 = (__int64)v30;
        v29[1] = (__int64)v31;
        v29[2] = (__int64)v32;
        v29[3] = v33;
        v29[4] = (__int64)v34;
        v29[5] = (__int64)v35;
        v29[6] = 0;
        v29[7] = v37;
        v29[8] = v38;
        v29[9] = v36;
      }
      else
      {
        v52 = v32 - v30;
        if ( v33 )
        {
          v60 = v52;
          j_j___libc_free_0(v33, &v35[-v33]);
          v52 = v60;
        }
        if ( v30 )
          j_j___libc_free_0(v30, v52);
      }
      *(_QWORD *)(v7 + 8) += 80LL;
      if ( *(_DWORD *)(v7 + 28) <= 2u )
        goto LABEL_45;
LABEL_61:
      if ( !(unsigned __int8)sub_EDD130(v7, &v63, (unsigned __int64)v16) )
      {
        v59 = *(_QWORD *)v7;
        v62 = *(_QWORD **)(v7 + 8);
        if ( *(_QWORD **)v7 != v62 )
        {
          v56 = v7;
          v45 = *(_QWORD **)v7;
          do
          {
            v46 = (_QWORD *)v45[6];
            if ( v46 )
            {
              v47 = v46 + 9;
              do
              {
                v48 = (_QWORD *)*(v47 - 3);
                v49 = (_QWORD *)*(v47 - 2);
                v47 -= 3;
                v50 = v48;
                if ( v49 != v48 )
                {
                  do
                  {
                    if ( *v50 )
                      j_j___libc_free_0(*v50, v50[2] - *v50);
                    v50 += 3;
                  }
                  while ( v49 != v50 );
                  v48 = (_QWORD *)*v47;
                }
                if ( v48 )
                  j_j___libc_free_0(v48, v47[2] - (_QWORD)v48);
              }
              while ( v46 != v47 );
              j_j___libc_free_0(v46, 72);
            }
            v51 = v45[3];
            if ( v51 )
              j_j___libc_free_0(v51, v45[5] - v51);
            if ( *v45 )
              j_j___libc_free_0(*v45, v45[2] - *v45);
            v45 += 10;
          }
          while ( v62 != v45 );
          *(_QWORD *)(v56 + 8) = v59;
        }
        goto LABEL_81;
      }
LABEL_45:
      v15 = v63;
      if ( v63 >= (unsigned int *)v16 )
      {
        v39 = v71;
        v57 = v73 - v71;
        goto LABEL_47;
      }
    }
  }
  v39 = 0;
LABEL_47:
  v5 = *(_QWORD *)v7;
LABEL_82:
  if ( v39 )
    j_j___libc_free_0(v39, v57);
  if ( v68 )
    j_j___libc_free_0(v68, v70 - v68);
  return v5;
}
