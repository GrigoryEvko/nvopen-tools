// Function: sub_EFFD30
// Address: 0xeffd30
//
__int64 __fastcall sub_EFFD30(__int64 a1, int *a2, __int64 a3)
{
  __int64 v6; // r8
  __int64 v7; // r9
  int v8; // edx
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // r14
  unsigned int v22; // esi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r10
  unsigned __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // r14
  unsigned int v32; // esi
  __int64 v33; // rbx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // r11
  unsigned __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rbx
  unsigned __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rbx
  unsigned __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rbx
  unsigned __int64 v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rbx
  int v54; // r10d
  __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // r14
  unsigned __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // r14
  unsigned __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // r14
  unsigned int v65; // esi
  __int64 v66; // rdx
  __int64 v67; // rcx
  int v68; // edi
  __int64 v69; // r14
  __int64 v70; // rdx
  unsigned __int64 v71; // rcx
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // [rsp-10h] [rbp-90h]
  __int64 v77; // [rsp-8h] [rbp-88h]
  unsigned __int8 v78; // [rsp+3h] [rbp-7Dh]
  unsigned int v79; // [rsp+4h] [rbp-7Ch]
  unsigned __int8 v80; // [rsp+4h] [rbp-7Ch]
  __int64 v81; // [rsp+8h] [rbp-78h]
  char v82; // [rsp+8h] [rbp-78h]
  unsigned int v83; // [rsp+8h] [rbp-78h]
  const void *v84; // [rsp+10h] [rbp-70h]
  __int64 v85; // [rsp+18h] [rbp-68h]
  __int64 v86; // [rsp+20h] [rbp-60h]
  __int64 v87; // [rsp+28h] [rbp-58h]
  unsigned int v88[20]; // [rsp+30h] [rbp-50h] BYREF

  v87 = a1 + 1576;
  sub_A19830(a1 + 1576, 9u, 4u);
  v8 = *(_DWORD *)(a1 + 1060);
  *(_DWORD *)(a1 + 1056) = 0;
  v85 = a1 + 1048;
  if ( v8 )
  {
    v9 = 0;
  }
  else
  {
    sub_C8D5F0(a1 + 1048, (const void *)(a1 + 1064), 1u, 8u, v6, v7);
    v9 = 8LL * *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v9) = 5;
  v10 = *(unsigned int *)(a1 + 1060);
  v11 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v11;
  v12 = *a2;
  if ( v11 + 1 > v10 )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v11 + 1, 8u, v11 + 1, v7);
    v11 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v11) = v12;
  ++*(_DWORD *)(a1 + 1056);
  sub_F02AE0(v88, a3, *((_QWORD *)a2 + 3), *((_QWORD *)a2 + 4));
  v14 = *(unsigned int *)(a1 + 1056);
  v15 = v88[0];
  if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1060) )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v14 + 1, 8u, v14 + 1, v13);
    v14 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v14) = v15;
  ++*(_DWORD *)(a1 + 1056);
  sub_F02AE0(v88, a3, *((_QWORD *)a2 + 1), *((_QWORD *)a2 + 2));
  v17 = *(unsigned int *)(a1 + 1056);
  v18 = v88[0];
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1060) )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v17 + 1, 8u, v17 + 1, v16);
    v17 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v17) = v18;
  ++*(_DWORD *)(a1 + 1056);
  sub_F02AE0(v88, a3, *((_QWORD *)a2 + 5), *((_QWORD *)a2 + 6));
  v20 = *(unsigned int *)(a1 + 1056);
  v21 = v88[0];
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1060) )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v20 + 1, 8u, v20 + 1, v19);
    v20 = *(unsigned int *)(a1 + 1056);
  }
  v22 = v88[0];
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v20) = v21;
  v23 = *(_QWORD *)(a1 + 1048);
  v24 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v24;
  sub_EFE900(v87, *(_DWORD *)(a1 + 1768), v23, v24, 0, 0, v22, 0);
  if ( !*((_BYTE *)a2 + 80) )
  {
    if ( !*((_BYTE *)a2 + 96) )
      goto LABEL_13;
    goto LABEL_41;
  }
  v54 = *(_DWORD *)(a1 + 1060);
  v55 = 0;
  *(_DWORD *)(a1 + 1056) = 0;
  if ( !v54 )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), 1u, 8u, v25, v26);
    v55 = 8LL * *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v55) = 6;
  ++*(_DWORD *)(a1 + 1056);
  sub_F02AE0(v88, a3, *((_QWORD *)a2 + 7), *((_QWORD *)a2 + 8));
  v57 = *(unsigned int *)(a1 + 1056);
  v58 = v88[0];
  if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1060) )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v57 + 1, 8u, v57 + 1, v56);
    v57 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v57) = v58;
  v59 = *(unsigned int *)(a1 + 1060);
  v60 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v60;
  v61 = (unsigned int)a2[18];
  if ( v60 + 1 > v59 )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v60 + 1, 8u, v60 + 1, v56);
    v60 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v60) = v61;
  v62 = *(unsigned int *)(a1 + 1060);
  v63 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v63;
  v64 = (unsigned int)a2[19];
  if ( v63 + 1 > v62 )
  {
    sub_C8D5F0(v85, (const void *)(a1 + 1064), v63 + 1, 8u, v63 + 1, v56);
    v63 = *(unsigned int *)(a1 + 1056);
  }
  v65 = v88[0];
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v63) = v64;
  v66 = *(_QWORD *)(a1 + 1048);
  v67 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v67;
  sub_EFE900(v87, *(_DWORD *)(a1 + 1776), v66, v67, 0, 0, v65, 0);
  v25 = v76;
  v26 = v77;
  if ( *((_BYTE *)a2 + 96) )
  {
LABEL_41:
    v68 = *(_DWORD *)(a1 + 1060);
    v69 = *((_QWORD *)a2 + 11);
    v70 = 0;
    *(_DWORD *)(a1 + 1056) = 0;
    if ( !v68 )
    {
      sub_C8D5F0(v85, (const void *)(a1 + 1064), 1u, 8u, v25, v26);
      v70 = 8LL * *(unsigned int *)(a1 + 1056);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v70) = 7;
    v71 = *(unsigned int *)(a1 + 1060);
    v72 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
    *(_DWORD *)(a1 + 1056) = v72;
    if ( v72 + 1 > v71 )
    {
      sub_C8D5F0(v85, (const void *)(a1 + 1064), v72 + 1, 8u, v72 + 1, v26);
      v72 = *(unsigned int *)(a1 + 1056);
    }
    v73 = v88[0];
    *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v72) = v69;
    v74 = *(_QWORD *)(a1 + 1048);
    v75 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
    *(_DWORD *)(a1 + 1056) = v75;
    sub_EFE900(v87, *(_DWORD *)(a1 + 1784), v74, v75, 0, 0, v73, 0);
  }
LABEL_13:
  v27 = *((_QWORD *)a2 + 13);
  v84 = (const void *)(a1 + 1064);
  v28 = (unsigned __int64)(unsigned int)a2[28] << 6;
  v86 = v27 + v28;
  if ( v27 != v27 + v28 )
  {
    v29 = a1;
    v30 = *((_QWORD *)a2 + 13);
    v31 = v29;
    do
    {
      *(_DWORD *)(v31 + 1056) = 0;
      sub_F02AE0(v88, a3, *(_QWORD *)v30, *(_QWORD *)(v30 + 8));
      v33 = v88[0];
      sub_F02AE0(v88, a3, *(_QWORD *)(v30 + 16), *(_QWORD *)(v30 + 24));
      v34 = *(unsigned __int8 *)(v30 + 56);
      v35 = v88[0];
      v36 = *(unsigned int *)(v31 + 1056);
      v37 = ((_BYTE)v34 == 0) + 8LL;
      if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(v31 + 1060) )
      {
        v78 = *(_BYTE *)(v30 + 56);
        v79 = v88[0];
        v81 = ((_BYTE)v34 == 0) + 8LL;
        sub_C8D5F0(v85, v84, v36 + 1, 8u, v34, v88[0]);
        v36 = *(unsigned int *)(v31 + 1056);
        v34 = v78;
        v35 = v79;
        v37 = v81;
      }
      *(_QWORD *)(*(_QWORD *)(v31 + 1048) + 8 * v36) = v37;
      v38 = *(unsigned int *)(v31 + 1060);
      v39 = (unsigned int)(*(_DWORD *)(v31 + 1056) + 1);
      *(_DWORD *)(v31 + 1056) = v39;
      if ( v39 + 1 > v38 )
      {
        v80 = v34;
        v83 = v35;
        sub_C8D5F0(v85, v84, v39 + 1, 8u, v34, v35);
        v39 = *(unsigned int *)(v31 + 1056);
        v34 = v80;
        v35 = v83;
      }
      *(_QWORD *)(*(_QWORD *)(v31 + 1048) + 8 * v39) = v33;
      v40 = (unsigned int)v35;
      v41 = *(unsigned int *)(v31 + 1060);
      v42 = (unsigned int)(*(_DWORD *)(v31 + 1056) + 1);
      *(_DWORD *)(v31 + 1056) = v42;
      if ( v42 + 1 > v41 )
      {
        v82 = v34;
        sub_C8D5F0(v85, v84, v42 + 1, 8u, v34, v35);
        v42 = *(unsigned int *)(v31 + 1056);
        LOBYTE(v34) = v82;
      }
      *(_QWORD *)(*(_QWORD *)(v31 + 1048) + 8 * v42) = v40;
      v43 = (unsigned int)(*(_DWORD *)(v31 + 1056) + 1);
      *(_DWORD *)(v31 + 1056) = v43;
      if ( (_BYTE)v34 )
      {
        sub_F02AE0(v88, a3, *(_QWORD *)(v30 + 32), *(_QWORD *)(v30 + 40));
        v45 = *(unsigned int *)(v31 + 1056);
        v46 = v88[0];
        if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(v31 + 1060) )
        {
          sub_C8D5F0(v85, v84, v45 + 1, 8u, v45 + 1, v44);
          v45 = *(unsigned int *)(v31 + 1056);
        }
        *(_QWORD *)(*(_QWORD *)(v31 + 1048) + 8 * v45) = v46;
        v47 = *(unsigned int *)(v31 + 1060);
        v48 = (unsigned int)(*(_DWORD *)(v31 + 1056) + 1);
        *(_DWORD *)(v31 + 1056) = v48;
        v49 = *(unsigned int *)(v30 + 48);
        if ( v48 + 1 > v47 )
        {
          sub_C8D5F0(v85, v84, v48 + 1, 8u, v48 + 1, v44);
          v48 = *(unsigned int *)(v31 + 1056);
        }
        *(_QWORD *)(*(_QWORD *)(v31 + 1048) + 8 * v48) = v49;
        v50 = *(unsigned int *)(v31 + 1060);
        v51 = (unsigned int)(*(_DWORD *)(v31 + 1056) + 1);
        *(_DWORD *)(v31 + 1056) = v51;
        v52 = *(unsigned int *)(v30 + 52);
        if ( v51 + 1 > v50 )
        {
          sub_C8D5F0(v85, v84, v51 + 1, 8u, v51 + 1, v44);
          v51 = *(unsigned int *)(v31 + 1056);
        }
        *(_QWORD *)(*(_QWORD *)(v31 + 1048) + 8 * v51) = v52;
        v32 = *(_DWORD *)(v31 + 1792);
        v43 = (unsigned int)(*(_DWORD *)(v31 + 1056) + 1);
        *(_DWORD *)(v31 + 1056) = v43;
      }
      else
      {
        v32 = *(_DWORD *)(v31 + 1800);
      }
      v30 += 64;
      sub_EFE900(v87, v32, *(_QWORD *)(v31 + 1048), v43, 0, 0, v88[0], 0);
    }
    while ( v86 != v30 );
  }
  return sub_A192A0(v87);
}
