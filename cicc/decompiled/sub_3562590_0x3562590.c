// Function: sub_3562590
// Address: 0x3562590
//
void __fastcall sub_3562590(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r15
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r12
  char *v13; // rcx
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // rax
  char *v17; // r14
  char *v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rax
  char *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rax
  char v32; // al
  __int64 v33; // rax
  unsigned int v34; // edx
  unsigned int v35; // eax
  bool v36; // cl
  unsigned int v37; // eax
  unsigned int v38; // edx
  int v39; // edx
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rax
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // r14
  __int64 v48; // r12
  __int64 i; // rbx
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rdx
  unsigned int v53; // ecx
  unsigned int v54; // edx
  bool v55; // di
  unsigned int v56; // edx
  unsigned int v57; // ecx
  int v58; // ecx
  int v59; // edx
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // r13
  __int64 v65; // rbx
  char *v66; // r12
  char v67; // dl
  __int64 v68; // rsi
  __int64 v69; // rdx
  __int64 v70; // [rsp+8h] [rbp-78h]
  __int64 v71; // [rsp+10h] [rbp-70h]
  __int64 v72; // [rsp+18h] [rbp-68h]
  __int64 v73; // [rsp+18h] [rbp-68h]
  int v74; // [rsp+18h] [rbp-68h]
  char *v75; // [rsp+20h] [rbp-60h]
  char *v76; // [rsp+20h] [rbp-60h]
  char *v77; // [rsp+20h] [rbp-60h]
  __int64 v78; // [rsp+20h] [rbp-60h]
  __int64 v79; // [rsp+20h] [rbp-60h]
  char *v81; // [rsp+30h] [rbp-50h]
  char *v83; // [rsp+40h] [rbp-40h]

  v7 = a6;
  v9 = (__int64)a1;
  v10 = a7;
  if ( a7 > a5 )
    v10 = a5;
  if ( a4 <= v10 )
  {
LABEL_19:
    v26 = sub_3553490(v9, a2, v7);
    if ( v7 != v26 && a3 != a2 )
    {
      v27 = v9 + 32;
      do
      {
        v34 = *(_DWORD *)(a2 + 52);
        v35 = *(_DWORD *)(v7 + 52);
        v36 = v34 > v35;
        if ( v34 == v35 )
        {
          v37 = *(_DWORD *)(a2 + 64);
          if ( !v37 || (v38 = *(_DWORD *)(v7 + 64), v37 == v38) || (v36 = v37 < v38, !v38) )
          {
            v39 = *(_DWORD *)(a2 + 56);
            v40 = *(_DWORD *)(v7 + 56);
            v36 = v39 < v40;
            if ( v39 == v40 )
              v36 = *(_DWORD *)(a2 + 60) > *(_DWORD *)(v7 + 60);
          }
        }
        v28 = *(_QWORD *)(v27 - 24);
        v29 = 8LL * *(unsigned int *)(v27 - 8);
        if ( v36 )
        {
          sub_C7D6A0(v28, v29, 8);
          *(_DWORD *)(v27 - 8) = 0;
          *(_QWORD *)(v27 - 24) = 0;
          *(_DWORD *)(v27 - 16) = 0;
          *(_DWORD *)(v27 - 12) = 0;
          ++*(_QWORD *)(v27 - 32);
          v30 = *(_QWORD *)(a2 + 8);
          ++*(_QWORD *)a2;
          v31 = *(_QWORD *)(v27 - 24);
          *(_QWORD *)(v27 - 24) = v30;
          LODWORD(v30) = *(_DWORD *)(a2 + 16);
          *(_QWORD *)(a2 + 8) = v31;
          LODWORD(v31) = *(_DWORD *)(v27 - 16);
          *(_DWORD *)(v27 - 16) = v30;
          LODWORD(v30) = *(_DWORD *)(a2 + 20);
          *(_DWORD *)(a2 + 16) = v31;
          LODWORD(v31) = *(_DWORD *)(v27 - 12);
          *(_DWORD *)(v27 - 12) = v30;
          LODWORD(v30) = *(_DWORD *)(a2 + 24);
          *(_DWORD *)(a2 + 20) = v31;
          LODWORD(v31) = *(_DWORD *)(v27 - 8);
          *(_DWORD *)(v27 - 8) = v30;
          *(_DWORD *)(a2 + 24) = v31;
          if ( v27 != a2 + 32 )
          {
            if ( *(_DWORD *)(a2 + 40) )
            {
              if ( *(_QWORD *)v27 != v27 + 16 )
                _libc_free(*(_QWORD *)v27);
              *(_QWORD *)v27 = *(_QWORD *)(a2 + 32);
              *(_DWORD *)(v27 + 8) = *(_DWORD *)(a2 + 40);
              *(_DWORD *)(v27 + 12) = *(_DWORD *)(a2 + 44);
              *(_QWORD *)(a2 + 32) = a2 + 48;
              *(_QWORD *)(a2 + 40) = 0;
            }
            else
            {
              *(_DWORD *)(v27 + 8) = 0;
            }
          }
          v32 = *(_BYTE *)(a2 + 48);
          a2 += 88;
          *(_BYTE *)(v27 + 16) = v32;
          *(_DWORD *)(v27 + 20) = *(_DWORD *)(a2 - 36);
          *(_DWORD *)(v27 + 24) = *(_DWORD *)(a2 - 32);
          *(_DWORD *)(v27 + 28) = *(_DWORD *)(a2 - 28);
          *(_DWORD *)(v27 + 32) = *(_DWORD *)(a2 - 24);
          *(_QWORD *)(v27 + 40) = *(_QWORD *)(a2 - 16);
          *(_DWORD *)(v27 + 48) = *(_DWORD *)(a2 - 8);
          v33 = v27 + 56;
          v27 += 88;
          if ( v26 == v7 )
            return;
        }
        else
        {
          sub_C7D6A0(v28, v29, 8);
          *(_DWORD *)(v27 - 8) = 0;
          *(_QWORD *)(v27 - 24) = 0;
          *(_DWORD *)(v27 - 16) = 0;
          *(_DWORD *)(v27 - 12) = 0;
          ++*(_QWORD *)(v27 - 32);
          v41 = *(_QWORD *)(v7 + 8);
          ++*(_QWORD *)v7;
          v42 = *(_QWORD *)(v27 - 24);
          *(_QWORD *)(v27 - 24) = v41;
          LODWORD(v41) = *(_DWORD *)(v7 + 16);
          *(_QWORD *)(v7 + 8) = v42;
          LODWORD(v42) = *(_DWORD *)(v27 - 16);
          *(_DWORD *)(v27 - 16) = v41;
          LODWORD(v41) = *(_DWORD *)(v7 + 20);
          *(_DWORD *)(v7 + 16) = v42;
          LODWORD(v42) = *(_DWORD *)(v27 - 12);
          *(_DWORD *)(v27 - 12) = v41;
          LODWORD(v41) = *(_DWORD *)(v7 + 24);
          *(_DWORD *)(v7 + 20) = v42;
          LODWORD(v42) = *(_DWORD *)(v27 - 8);
          *(_DWORD *)(v27 - 8) = v41;
          *(_DWORD *)(v7 + 24) = v42;
          if ( v27 != v7 + 32 )
          {
            if ( *(_DWORD *)(v7 + 40) )
            {
              if ( *(_QWORD *)v27 != v27 + 16 )
                _libc_free(*(_QWORD *)v27);
              *(_QWORD *)v27 = *(_QWORD *)(v7 + 32);
              *(_DWORD *)(v27 + 8) = *(_DWORD *)(v7 + 40);
              *(_DWORD *)(v27 + 12) = *(_DWORD *)(v7 + 44);
              *(_QWORD *)(v7 + 32) = v7 + 48;
              *(_QWORD *)(v7 + 40) = 0;
            }
            else
            {
              *(_DWORD *)(v27 + 8) = 0;
            }
          }
          v43 = *(_BYTE *)(v7 + 48);
          v7 += 88;
          *(_BYTE *)(v27 + 16) = v43;
          *(_DWORD *)(v27 + 20) = *(_DWORD *)(v7 - 36);
          *(_DWORD *)(v27 + 24) = *(_DWORD *)(v7 - 32);
          *(_DWORD *)(v27 + 28) = *(_DWORD *)(v7 - 28);
          *(_DWORD *)(v27 + 32) = *(_DWORD *)(v7 - 24);
          *(_QWORD *)(v27 + 40) = *(_QWORD *)(v7 - 16);
          *(_DWORD *)(v27 + 48) = *(_DWORD *)(v7 - 8);
          v33 = v27 + 56;
          v27 += 88;
          if ( v26 == v7 )
            return;
        }
      }
      while ( a3 != a2 );
      v9 = v33;
    }
    if ( v26 != v7 )
      sub_3553490(v7, v26, v9);
    return;
  }
  v11 = a5;
  if ( a7 >= a5 )
    goto LABEL_46;
  v83 = a1;
  v13 = (char *)a2;
  v14 = a4;
  if ( a4 <= a5 )
    goto LABEL_14;
LABEL_6:
  v75 = v13;
  v15 = v14 / 2;
  v16 = 5 * (v14 / 2);
  v14 -= v14 / 2;
  v17 = &v83[16 * v16 + 8 * v15];
  v18 = (char *)sub_353ECD0(v13, a3, v17);
  v21 = (__int64)v75;
  v81 = v18;
  v22 = 0x2E8BA2E8BA2E8BA3LL * ((v18 - v75) >> 3);
  if ( v14 <= v22 )
    goto LABEL_15;
LABEL_7:
  if ( a7 >= v22 )
  {
    v23 = (__int64)v17;
    if ( v22 )
    {
      v73 = v21;
      v79 = sub_3553490(v21, (__int64)v81, a6);
      v62 = v73 - 56;
      v63 = 0x2E8BA2E8BA2E8BA3LL * ((v73 - (__int64)v17) >> 3);
      if ( v73 - (__int64)v17 > 0 )
      {
        v74 = v15;
        v64 = v63;
        v71 = v22;
        v65 = v62;
        v70 = v11;
        v66 = v81 - 56;
        do
        {
          sub_C7D6A0(*((_QWORD *)v66 - 3), 8LL * *((unsigned int *)v66 - 2), 8);
          ++*((_QWORD *)v66 - 4);
          *((_DWORD *)v66 - 2) = 0;
          *((_QWORD *)v66 - 3) = 0;
          *((_DWORD *)v66 - 4) = 0;
          *((_DWORD *)v66 - 3) = 0;
          v68 = *(_QWORD *)(v65 - 24);
          ++*(_QWORD *)(v65 - 32);
          v69 = *((_QWORD *)v66 - 3);
          *((_QWORD *)v66 - 3) = v68;
          LODWORD(v68) = *(_DWORD *)(v65 - 16);
          *(_QWORD *)(v65 - 24) = v69;
          LODWORD(v69) = *((_DWORD *)v66 - 4);
          *((_DWORD *)v66 - 4) = v68;
          LODWORD(v68) = *(_DWORD *)(v65 - 12);
          *(_DWORD *)(v65 - 16) = v69;
          LODWORD(v69) = *((_DWORD *)v66 - 3);
          *((_DWORD *)v66 - 3) = v68;
          LODWORD(v68) = *(_DWORD *)(v65 - 8);
          *(_DWORD *)(v65 - 12) = v69;
          LODWORD(v69) = *((_DWORD *)v66 - 2);
          *((_DWORD *)v66 - 2) = v68;
          *(_DWORD *)(v65 - 8) = v69;
          if ( v66 != (char *)v65 )
          {
            if ( *(_DWORD *)(v65 + 8) )
            {
              if ( *(char **)v66 != v66 + 16 )
                _libc_free(*(_QWORD *)v66);
              *(_QWORD *)v66 = *(_QWORD *)v65;
              *((_DWORD *)v66 + 2) = *(_DWORD *)(v65 + 8);
              *((_DWORD *)v66 + 3) = *(_DWORD *)(v65 + 12);
              *(_QWORD *)v65 = v65 + 16;
              *(_DWORD *)(v65 + 12) = 0;
              *(_DWORD *)(v65 + 8) = 0;
            }
            else
            {
              *((_DWORD *)v66 + 2) = 0;
            }
          }
          v67 = *(_BYTE *)(v65 + 16);
          v66 -= 88;
          v65 -= 88;
          v66[104] = v67;
          *((_DWORD *)v66 + 27) = *(_DWORD *)(v65 + 108);
          *((_DWORD *)v66 + 28) = *(_DWORD *)(v65 + 112);
          *((_DWORD *)v66 + 29) = *(_DWORD *)(v65 + 116);
          *((_DWORD *)v66 + 30) = *(_DWORD *)(v65 + 120);
          *((_QWORD *)v66 + 16) = *(_QWORD *)(v65 + 128);
          *((_DWORD *)v66 + 34) = *(_DWORD *)(v65 + 136);
          --v64;
        }
        while ( v64 );
        LODWORD(v15) = v74;
        v22 = v71;
        v11 = v70;
      }
      v23 = sub_3553490(a6, v79, (__int64)v17);
    }
    goto LABEL_9;
  }
  while ( 1 )
  {
LABEL_15:
    v23 = (__int64)v81;
    if ( a7 < v14 )
    {
      v23 = sub_3541720((__int64)v17, v21, (__int64)v81, v21, v19, v20);
    }
    else if ( v14 )
    {
      v72 = v21;
      v78 = sub_3553490((__int64)v17, v21, a6);
      sub_3553490(v72, (__int64)v81, (__int64)v17);
      v23 = sub_35623D0(a6, v78, (__int64)v81);
    }
LABEL_9:
    v76 = (char *)v23;
    v11 -= v22;
    sub_3562590((_DWORD)v83, (_DWORD)v17, v23, v15, v22, a6, a7);
    v24 = a7;
    if ( a7 > v11 )
      v24 = v11;
    if ( v24 >= v14 )
    {
      v7 = a6;
      a2 = (__int64)v81;
      v9 = (__int64)v76;
      goto LABEL_19;
    }
    if ( a7 >= v11 )
      break;
    v83 = v76;
    v13 = v81;
    if ( v14 > v11 )
      goto LABEL_6;
LABEL_14:
    v77 = v13;
    v22 = v11 / 2;
    v81 = &v13[88 * (v11 / 2)];
    v25 = (char *)sub_353EC20(v83, (__int64)v13, v81);
    v21 = (__int64)v77;
    v17 = v25;
    v15 = 0x2E8BA2E8BA2E8BA3LL * ((v25 - v83) >> 3);
    v14 -= v15;
    if ( v14 > v11 / 2 )
      goto LABEL_7;
  }
  v7 = a6;
  a2 = (__int64)v81;
  v9 = (__int64)v76;
LABEL_46:
  v44 = sub_3553490(a2, a3, v7);
  v45 = a3;
  v46 = v44;
  if ( a2 == v9 )
    goto LABEL_88;
  if ( v7 != v44 )
  {
    v47 = a2 - 88;
    v48 = v44 - 88;
    for ( i = a3 - 56; ; i -= 88 )
    {
      v53 = *(_DWORD *)(v48 + 52);
      v54 = *(_DWORD *)(v47 + 52);
      v55 = v53 > v54;
      if ( v53 == v54 )
      {
        v56 = *(_DWORD *)(v48 + 64);
        if ( !v56 || (v57 = *(_DWORD *)(v47 + 64)) == 0 || (v55 = v56 < v57, v56 == v57) )
        {
          v58 = *(_DWORD *)(v48 + 56);
          v59 = *(_DWORD *)(v47 + 56);
          v55 = v58 < v59;
          if ( v58 == v59 )
            v55 = *(_DWORD *)(v48 + 60) > *(_DWORD *)(v47 + 60);
        }
      }
      v50 = 8LL * *(unsigned int *)(i - 8);
      if ( v55 )
      {
        sub_C7D6A0(*(_QWORD *)(i - 24), v50, 8);
        *(_DWORD *)(i - 8) = 0;
        *(_QWORD *)(i - 24) = 0;
        *(_DWORD *)(i - 16) = 0;
        *(_DWORD *)(i - 12) = 0;
        ++*(_QWORD *)(i - 32);
        v51 = *(_QWORD *)(v47 + 8);
        ++*(_QWORD *)v47;
        v52 = *(_QWORD *)(i - 24);
        *(_QWORD *)(i - 24) = v51;
        LODWORD(v51) = *(_DWORD *)(v47 + 16);
        *(_QWORD *)(v47 + 8) = v52;
        LODWORD(v52) = *(_DWORD *)(i - 16);
        *(_DWORD *)(i - 16) = v51;
        LODWORD(v51) = *(_DWORD *)(v47 + 20);
        *(_DWORD *)(v47 + 16) = v52;
        LODWORD(v52) = *(_DWORD *)(i - 12);
        *(_DWORD *)(i - 12) = v51;
        LODWORD(v51) = *(_DWORD *)(v47 + 24);
        *(_DWORD *)(v47 + 20) = v52;
        LODWORD(v52) = *(_DWORD *)(i - 8);
        *(_DWORD *)(i - 8) = v51;
        *(_DWORD *)(v47 + 24) = v52;
        if ( i != v47 + 32 )
        {
          if ( *(_DWORD *)(v47 + 40) )
          {
            if ( *(_QWORD *)i != i + 16 )
              _libc_free(*(_QWORD *)i);
            *(_QWORD *)i = *(_QWORD *)(v47 + 32);
            *(_DWORD *)(i + 8) = *(_DWORD *)(v47 + 40);
            *(_DWORD *)(i + 12) = *(_DWORD *)(v47 + 44);
            *(_QWORD *)(v47 + 32) = v47 + 48;
            *(_QWORD *)(v47 + 40) = 0;
          }
          else
          {
            *(_DWORD *)(i + 8) = 0;
          }
        }
        *(_BYTE *)(i + 16) = *(_BYTE *)(v47 + 48);
        *(_DWORD *)(i + 20) = *(_DWORD *)(v47 + 52);
        *(_DWORD *)(i + 24) = *(_DWORD *)(v47 + 56);
        *(_DWORD *)(i + 28) = *(_DWORD *)(v47 + 60);
        *(_DWORD *)(i + 32) = *(_DWORD *)(v47 + 64);
        *(_QWORD *)(i + 40) = *(_QWORD *)(v47 + 72);
        *(_DWORD *)(i + 48) = *(_DWORD *)(v47 + 80);
        if ( v47 == v9 )
        {
          v45 = i - 32;
          v46 = v48 + 88;
LABEL_88:
          sub_35623D0(v7, v46, v45);
          return;
        }
        v47 -= 88;
      }
      else
      {
        sub_C7D6A0(*(_QWORD *)(i - 24), v50, 8);
        *(_DWORD *)(i - 8) = 0;
        *(_QWORD *)(i - 24) = 0;
        *(_DWORD *)(i - 16) = 0;
        *(_DWORD *)(i - 12) = 0;
        ++*(_QWORD *)(i - 32);
        v60 = *(_QWORD *)(v48 + 8);
        ++*(_QWORD *)v48;
        v61 = *(_QWORD *)(i - 24);
        *(_QWORD *)(i - 24) = v60;
        LODWORD(v60) = *(_DWORD *)(v48 + 16);
        *(_QWORD *)(v48 + 8) = v61;
        LODWORD(v61) = *(_DWORD *)(i - 16);
        *(_DWORD *)(i - 16) = v60;
        LODWORD(v60) = *(_DWORD *)(v48 + 20);
        *(_DWORD *)(v48 + 16) = v61;
        LODWORD(v61) = *(_DWORD *)(i - 12);
        *(_DWORD *)(i - 12) = v60;
        LODWORD(v60) = *(_DWORD *)(v48 + 24);
        *(_DWORD *)(v48 + 20) = v61;
        LODWORD(v61) = *(_DWORD *)(i - 8);
        *(_DWORD *)(i - 8) = v60;
        *(_DWORD *)(v48 + 24) = v61;
        if ( i != v48 + 32 )
        {
          if ( *(_DWORD *)(v48 + 40) )
          {
            if ( *(_QWORD *)i != i + 16 )
              _libc_free(*(_QWORD *)i);
            *(_QWORD *)i = *(_QWORD *)(v48 + 32);
            *(_DWORD *)(i + 8) = *(_DWORD *)(v48 + 40);
            *(_DWORD *)(i + 12) = *(_DWORD *)(v48 + 44);
            *(_QWORD *)(v48 + 32) = v48 + 48;
            *(_QWORD *)(v48 + 40) = 0;
          }
          else
          {
            *(_DWORD *)(i + 8) = 0;
          }
        }
        *(_BYTE *)(i + 16) = *(_BYTE *)(v48 + 48);
        *(_DWORD *)(i + 20) = *(_DWORD *)(v48 + 52);
        *(_DWORD *)(i + 24) = *(_DWORD *)(v48 + 56);
        *(_DWORD *)(i + 28) = *(_DWORD *)(v48 + 60);
        *(_DWORD *)(i + 32) = *(_DWORD *)(v48 + 64);
        *(_QWORD *)(i + 40) = *(_QWORD *)(v48 + 72);
        *(_DWORD *)(i + 48) = *(_DWORD *)(v48 + 80);
        if ( v7 == v48 )
          return;
        v48 -= 88;
      }
    }
  }
}
