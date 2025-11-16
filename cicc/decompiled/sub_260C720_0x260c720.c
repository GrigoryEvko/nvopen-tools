// Function: sub_260C720
// Address: 0x260c720
//
void __fastcall sub_260C720(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  unsigned __int64 *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rbx
  char *v10; // r14
  char *v11; // rax
  __int64 v12; // r11
  unsigned __int64 v13; // rcx
  char *v14; // r10
  signed __int64 v15; // r8
  __int64 v16; // r11
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rax
  unsigned __int64 *v19; // r14
  unsigned __int64 *v20; // r13
  unsigned __int64 *j; // r12
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rdi
  unsigned __int64 *v28; // r14
  unsigned __int64 *v29; // r12
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // r9
  unsigned __int64 v32; // rbx
  unsigned __int64 v33; // r15
  __int64 v34; // rsi
  __int64 v35; // rdi
  unsigned __int64 v36; // r15
  __int64 v37; // rsi
  __int64 v38; // rdi
  char *v39; // rax
  unsigned __int64 v40; // r15
  __int64 v41; // rsi
  __int64 v42; // rdi
  unsigned __int64 *v43; // r14
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // rbx
  unsigned __int64 v46; // r13
  __int64 v47; // rsi
  __int64 v48; // rdi
  signed __int64 v49; // [rsp+8h] [rbp-68h]
  signed __int64 v50; // [rsp+8h] [rbp-68h]
  char *v51; // [rsp+10h] [rbp-60h]
  int v52; // [rsp+10h] [rbp-60h]
  int v53; // [rsp+10h] [rbp-60h]
  signed __int64 v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v56; // [rsp+18h] [rbp-58h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  __int64 v58; // [rsp+18h] [rbp-58h]
  int v59; // [rsp+18h] [rbp-58h]
  signed __int64 v60; // [rsp+20h] [rbp-50h]
  unsigned __int64 *i; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v62; // [rsp+20h] [rbp-50h]
  __int64 v63; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v64; // [rsp+20h] [rbp-50h]
  __int64 v65; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v66; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v67; // [rsp+30h] [rbp-40h]
  unsigned __int64 v68; // [rsp+30h] [rbp-40h]
  __int64 v69; // [rsp+38h] [rbp-38h]
  unsigned __int64 *v70; // [rsp+38h] [rbp-38h]
  unsigned __int64 v71; // [rsp+38h] [rbp-38h]
  unsigned __int64 v72; // [rsp+38h] [rbp-38h]
  char *v73; // [rsp+38h] [rbp-38h]
  char *v74; // [rsp+38h] [rbp-38h]
  char *v75; // [rsp+38h] [rbp-38h]
  char *v76; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v66 = a1;
    v67 = (unsigned __int64 *)a3;
    v8 = a7;
    if ( a5 <= a7 )
      v8 = a5;
    if ( a4 <= v8 )
    {
      v28 = a2;
      v29 = a1;
      for ( i = sub_260C5C0(a1, (__int64)a2, a6); i != v7; v29 += 3 )
      {
        if ( v28 == v67 )
        {
          sub_260C5C0(v7, (__int64)i, v29);
          return;
        }
        v30 = *v7;
        v31 = *v29;
        v32 = v29[1];
        v72 = *v29;
        if ( *(unsigned int *)(*v28 + 4) * 0x86BCA1AF286BCA1BLL * ((__int64)(v28[1] - *v28) >> 3) <= *(unsigned int *)(v30 + 4) * 0x86BCA1AF286BCA1BLL * ((__int64)(v7[1] - v30) >> 3) )
        {
          *v29 = v30;
          v29[1] = v7[1];
          v29[2] = v7[2];
          *v7 = 0;
          v7[1] = 0;
          v36 = v72;
          v7[2] = 0;
          while ( v36 != v32 )
          {
            v37 = *(unsigned int *)(v36 + 144);
            v38 = *(_QWORD *)(v36 + 128);
            v36 += 152LL;
            sub_C7D6A0(v38, 8 * v37, 4);
            sub_C7D6A0(*(_QWORD *)(v36 - 56), 8LL * *(unsigned int *)(v36 - 40), 4);
            sub_C7D6A0(*(_QWORD *)(v36 - 88), 16LL * *(unsigned int *)(v36 - 72), 8);
            sub_C7D6A0(*(_QWORD *)(v36 - 120), 16LL * *(unsigned int *)(v36 - 104), 8);
          }
          if ( v72 )
            j_j___libc_free_0(v72);
          v7 += 3;
        }
        else
        {
          *v29 = *v28;
          v33 = v31;
          v29[1] = v28[1];
          v29[2] = v28[2];
          *v28 = 0;
          v28[1] = 0;
          v28[2] = 0;
          while ( v33 != v32 )
          {
            v34 = *(unsigned int *)(v33 + 144);
            v35 = *(_QWORD *)(v33 + 128);
            v33 += 152LL;
            sub_C7D6A0(v35, 8 * v34, 4);
            sub_C7D6A0(*(_QWORD *)(v33 - 56), 8LL * *(unsigned int *)(v33 - 40), 4);
            sub_C7D6A0(*(_QWORD *)(v33 - 88), 16LL * *(unsigned int *)(v33 - 72), 8);
            sub_C7D6A0(*(_QWORD *)(v33 - 120), 16LL * *(unsigned int *)(v33 - 104), 8);
          }
          if ( v72 )
            j_j___libc_free_0(v72);
          v28 += 3;
        }
      }
      return;
    }
    v9 = a5;
    if ( a5 <= a7 )
      break;
    if ( a4 <= a5 )
    {
      v63 = a5 / 2;
      v74 = (char *)&a2[a5 / 2 + ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v39 = (char *)sub_25F6920(a1, (__int64)a2, v74);
      v15 = v63;
      v14 = v74;
      v10 = v39;
      v13 = 0xAAAAAAAAAAAAAAABLL * ((v39 - (char *)a1) >> 3);
    }
    else
    {
      v69 = a4 / 2;
      v10 = (char *)&a1[a4 / 2 + ((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v11 = (char *)sub_25F69C0(a2, a3, v10);
      v13 = v69;
      v14 = v11;
      v15 = 0xAAAAAAAAAAAAAAABLL * ((v11 - (char *)a2) >> 3);
    }
    v16 = v12 - v13;
    if ( v16 <= v15 || v15 > a7 )
    {
      if ( v16 > a7 )
      {
        v54 = v15;
        v59 = v13;
        v65 = v16;
        v76 = v14;
        v17 = (unsigned __int64 *)sub_25F77F0(v10, (char *)a2, v14);
        v15 = v54;
        LODWORD(v13) = v59;
        v16 = v65;
        v14 = v76;
      }
      else
      {
        v17 = (unsigned __int64 *)v14;
        if ( v16 )
        {
          v49 = v15;
          v52 = v13;
          v57 = v16;
          v73 = v14;
          v62 = sub_260C5C0((unsigned __int64 *)v10, (__int64)a2, v7);
          sub_260C5C0(a2, (__int64)v73, (unsigned __int64 *)v10);
          v17 = sub_260C430((__int64)v7, v62, v73);
          v14 = v73;
          v16 = v57;
          LODWORD(v13) = v52;
          v15 = v49;
        }
      }
    }
    else
    {
      v17 = (unsigned __int64 *)v10;
      if ( v15 )
      {
        v50 = v15;
        v53 = v13;
        v58 = v16;
        v75 = v14;
        v64 = sub_260C5C0(a2, (__int64)v14, v7);
        sub_260C430((__int64)v10, a2, v75);
        v17 = sub_260C5C0(v7, (__int64)v64, (unsigned __int64 *)v10);
        v14 = v75;
        v16 = v58;
        LODWORD(v13) = v53;
        v15 = v50;
      }
    }
    v51 = v14;
    v55 = v16;
    v60 = v15;
    v70 = v17;
    sub_260C720((_DWORD)a1, (_DWORD)v10, (_DWORD)v17, v13, v15, (_DWORD)v7, a7);
    a6 = v7;
    a4 = v55;
    a3 = (__int64)v67;
    a5 = v9 - v60;
    a2 = (unsigned __int64 *)v51;
    a1 = v70;
  }
  v18 = sub_260C5C0(a2, a3, a6);
  if ( a2 == a1 )
  {
    sub_260C430((__int64)v7, v18, v67);
    return;
  }
  if ( v18 != v7 )
  {
    v56 = v7;
    v19 = v18 - 3;
    v20 = a2 - 3;
    for ( j = v67 - 3; ; j -= 3 )
    {
      v22 = *v20;
      v23 = *j;
      v24 = j[1];
      v71 = *j;
      if ( *(unsigned int *)(*v19 + 4) * 0x86BCA1AF286BCA1BLL * ((__int64)(v19[1] - *v19) >> 3) <= *(unsigned int *)(v22 + 4)
                                                                                                 * 0x86BCA1AF286BCA1BLL
                                                                                                 * ((__int64)(v20[1] - v22) >> 3) )
      {
        *j = *v19;
        j[1] = v19[1];
        j[2] = v19[2];
        *v19 = 0;
        v19[1] = 0;
        v40 = v71;
        v19[2] = 0;
        while ( v24 != v40 )
        {
          v41 = *(unsigned int *)(v40 + 144);
          v42 = *(_QWORD *)(v40 + 128);
          v40 += 152LL;
          sub_C7D6A0(v42, 8 * v41, 4);
          sub_C7D6A0(*(_QWORD *)(v40 - 56), 8LL * *(unsigned int *)(v40 - 40), 4);
          sub_C7D6A0(*(_QWORD *)(v40 - 88), 16LL * *(unsigned int *)(v40 - 72), 8);
          sub_C7D6A0(*(_QWORD *)(v40 - 120), 16LL * *(unsigned int *)(v40 - 104), 8);
        }
        if ( v71 )
          j_j___libc_free_0(v71);
        if ( v56 == v19 )
          return;
        v19 -= 3;
      }
      else
      {
        *j = v22;
        v25 = v23;
        j[1] = v20[1];
        j[2] = v20[2];
        *v20 = 0;
        v20[1] = 0;
        v20[2] = 0;
        while ( v25 != v24 )
        {
          v26 = *(unsigned int *)(v25 + 144);
          v27 = *(_QWORD *)(v25 + 128);
          v25 += 152LL;
          sub_C7D6A0(v27, 8 * v26, 4);
          sub_C7D6A0(*(_QWORD *)(v25 - 56), 8LL * *(unsigned int *)(v25 - 40), 4);
          sub_C7D6A0(*(_QWORD *)(v25 - 88), 16LL * *(unsigned int *)(v25 - 72), 8);
          sub_C7D6A0(*(_QWORD *)(v25 - 120), 16LL * *(unsigned int *)(v25 - 104), 8);
        }
        if ( v71 )
          j_j___libc_free_0(v71);
        if ( v20 == v66 )
        {
          v43 = v19 + 3;
          if ( (char *)v43 - (char *)v56 > 0 )
          {
            v68 = 0xAAAAAAAAAAAAAAABLL * (v43 - v56);
            do
            {
              v43 -= 3;
              v44 = *(j - 3);
              j -= 3;
              v45 = j[1];
              v46 = v44;
              *j = *v43;
              j[1] = v43[1];
              j[2] = v43[2];
              *v43 = 0;
              v43[1] = 0;
              v43[2] = 0;
              while ( v45 != v46 )
              {
                v47 = *(unsigned int *)(v46 + 144);
                v48 = *(_QWORD *)(v46 + 128);
                v46 += 152LL;
                sub_C7D6A0(v48, 8 * v47, 4);
                sub_C7D6A0(*(_QWORD *)(v46 - 56), 8LL * *(unsigned int *)(v46 - 40), 4);
                sub_C7D6A0(*(_QWORD *)(v46 - 88), 16LL * *(unsigned int *)(v46 - 72), 8);
                sub_C7D6A0(*(_QWORD *)(v46 - 120), 16LL * *(unsigned int *)(v46 - 104), 8);
              }
              if ( v44 )
                j_j___libc_free_0(v44);
              --v68;
            }
            while ( v68 );
          }
          return;
        }
        v20 -= 3;
      }
    }
  }
}
