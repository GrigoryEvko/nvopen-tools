// Function: sub_331BE70
// Address: 0x331be70
//
unsigned __int64 __fastcall sub_331BE70(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned int *v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rsi
  int v9; // ebx
  __int64 *v10; // r9
  unsigned int v11; // r15d
  unsigned __int64 result; // rax
  bool v13; // bl
  int v14; // esi
  __int64 v15; // rdx
  unsigned __int64 *v16; // rcx
  int v17; // esi
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned int v24; // r9d
  __int64 v25; // r15
  unsigned int v26; // eax
  unsigned int v27; // ecx
  __int64 v28; // rdx
  unsigned int v29; // r10d
  __int64 v30; // rax
  __int64 v31; // r8
  int v32; // r10d
  unsigned int v33; // r9d
  char v34; // dl
  unsigned __int64 v35; // rax
  int v36; // edx
  unsigned int v37; // ecx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rdx
  unsigned int v42; // r14d
  __int64 v43; // rbx
  int v44; // r9d
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  unsigned __int64 *v49; // rax
  unsigned int v50; // r12d
  __int64 v51; // rax
  unsigned int v52; // edx
  unsigned int v53; // eax
  __int64 v54; // rsi
  unsigned __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // [rsp+8h] [rbp-B8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  unsigned int v59; // [rsp+20h] [rbp-A0h]
  char v60; // [rsp+20h] [rbp-A0h]
  __int64 v61; // [rsp+20h] [rbp-A0h]
  unsigned int v62; // [rsp+28h] [rbp-98h]
  int v63; // [rsp+28h] [rbp-98h]
  int v64; // [rsp+30h] [rbp-90h]
  unsigned int v65; // [rsp+30h] [rbp-90h]
  __int64 v66; // [rsp+38h] [rbp-88h]
  unsigned int v67; // [rsp+38h] [rbp-88h]
  unsigned int v68; // [rsp+40h] [rbp-80h]
  unsigned int v69; // [rsp+40h] [rbp-80h]
  __int64 v70; // [rsp+40h] [rbp-80h]
  __int64 *v71; // [rsp+40h] [rbp-80h]
  unsigned int v73[4]; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v74[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v75[10]; // [rsp+70h] [rbp-50h] BYREF

  v4 = a1 + 2;
  v5 = (__int64)a1;
  v6 = a1[4];
  if ( !v6 || (v7 = *((_QWORD *)a1 + 1), *(_DWORD *)(v7 + 12) >= *(_DWORD *)(v7 + 8)) )
  {
    v18 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 136LL);
    sub_F03AD0(v4, *(_DWORD *)(*(_QWORD *)a1 + 136LL));
    ++*(_DWORD *)(*((_QWORD *)a1 + 1) + v18 + 12);
    v7 = *((_QWORD *)a1 + 1);
    v6 = a1[4];
  }
  v8 = v7 + 16LL * v6 - 16;
  v9 = *(_DWORD *)(v8 + 12);
  v10 = *(__int64 **)v8;
  if ( !v9 && *v10 > a2 )
  {
    v19 = sub_F03A30((__int64 *)v4, v6 - 1);
    if ( v19 )
    {
      v20 = (__int64 *)(16 * (v19 & 0x3F) + (v19 & 0xFFFFFFFFFFFFFFC0LL));
      v21 = a1[4];
      v8 = *((_QWORD *)a1 + 1) + 16 * v21 - 16;
      v10 = *(__int64 **)v8;
      if ( a2 != v20[1] )
      {
        v9 = *(_DWORD *)(v8 + 12);
        goto LABEL_5;
      }
      v71 = *(__int64 **)v8;
      sub_F03AD0(v4, v21 - 1);
      result = a3;
      if ( *v71 > a3 )
      {
        v20[1] = a3;
        v17 = a1[4] - 1;
        if ( a1[4] != 1 )
          return sub_325DE80(v5, v17, a3);
        return result;
      }
      a2 = *v20;
      sub_32B0EF0((__int64)a1, 0);
    }
    else
    {
      **(_QWORD **)a1 = a2;
    }
    v8 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
    v9 = *(_DWORD *)(v8 + 12);
    v10 = *(__int64 **)v8;
  }
LABEL_5:
  v11 = *(_DWORD *)(v8 + 8);
  result = sub_325E8E0((__int64)v10, (unsigned int *)(v8 + 12), v11, a2, a3);
  v13 = v11 == v9;
  if ( (unsigned int)result <= 0xB )
    goto LABEL_6;
  v22 = a1[4] - 1;
  v68 = *(_DWORD *)(*((_QWORD *)a1 + 1) + 16 * v22 + 12);
  v23 = sub_F03A30((__int64 *)v4, a1[4] - 1);
  v24 = v68;
  v25 = v23;
  if ( v23 )
  {
    v69 = 2;
    v26 = (v23 & 0x3F) + 1;
    v75[0] = v25 & 0xFFFFFFFFFFFFFFC0LL;
    v27 = 1;
    v73[0] = v26;
    v24 += v26;
  }
  else
  {
    v69 = 1;
    v26 = 0;
    v27 = 0;
  }
  v28 = *((_QWORD *)a1 + 1) + 16 * v22;
  v29 = *(_DWORD *)(v28 + 8);
  v59 = v24;
  v62 = v27;
  v73[v27] = v29;
  v64 = v26 + v29;
  v75[v27] = *(_QWORD *)v28;
  v66 = v27;
  v30 = sub_F03C90((__int64 *)v4, v22);
  v31 = v66;
  v32 = v64;
  v33 = v59;
  if ( v30 )
  {
    v34 = v30;
    v35 = v30 & 0xFFFFFFFFFFFFFFC0LL;
    v31 = v69;
    v67 = v62 + 2;
    v36 = (v34 & 0x3F) + 1;
    v75[v69] = v35;
    v32 = v36 + v64;
    v37 = 22;
    v73[v69] = v36;
    if ( v62 )
      v37 = 33;
    if ( v32 + 1 > v37 )
      goto LABEL_23;
LABEL_47:
    v60 = 0;
    v69 = 0;
    goto LABEL_28;
  }
  v53 = 11;
  if ( v69 != 1 )
    v53 = 22;
  if ( v64 + 1 <= v53 )
  {
    v67 = v69;
    goto LABEL_47;
  }
  if ( v69 != 1 )
  {
    v36 = v73[v66];
    v35 = v75[v66];
    v69 = v62;
    v67 = 2;
LABEL_23:
    v38 = v67++;
    goto LABEL_24;
  }
  v36 = v73[1];
  v35 = v75[1];
  v38 = 1;
  v67 = 2;
  v31 = 1;
LABEL_24:
  v75[v38] = v35;
  v39 = *(_QWORD *)a1;
  v73[v38] = v36;
  v73[v31] = 0;
  v40 = *(_QWORD *)(v39 + 144);
  v41 = *(_QWORD **)v40;
  if ( *(_QWORD *)v40 )
  {
    *(_QWORD *)v40 = *v41;
  }
  else
  {
    v54 = *(_QWORD *)(v40 + 8);
    *(_QWORD *)(v40 + 88) += 192LL;
    v55 = (v54 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_QWORD *)(v40 + 16) >= v55 + 192 && v54 )
    {
      *(_QWORD *)(v40 + 8) = v55 + 192;
      if ( !v55 )
        goto LABEL_27;
      v41 = (_QWORD *)((v54 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    }
    else
    {
      v61 = v31;
      v63 = v32;
      v65 = v33;
      v56 = sub_9D1E70(v40 + 8, 192, 192, 6);
      v31 = v61;
      v32 = v63;
      v33 = v65;
      v41 = (_QWORD *)v56;
    }
  }
  memset(v41, 0, 0xC0u);
LABEL_27:
  v75[v31] = v41;
  v60 = 1;
LABEL_28:
  v58 = sub_F03E60(v67, v32, 11, (__int64)v73, (__int64)v74, v33, 1u);
  sub_32B1120((__int64)v75, v67, v73, (__int64)v74);
  if ( v25 )
    sub_F03AD0(v4, v22);
  v57 = a2;
  v42 = v22;
  v43 = 0;
  while ( 1 )
  {
    v44 = v74[v43];
    v45 = (unsigned int)(v44 - 1);
    v46 = v75[v43];
    v47 = *(_QWORD *)(v46 + 16 * v45 + 8);
    if ( v69 == (_DWORD)v43 && v60 )
    {
      v42 += (unsigned __int8)sub_331BBE0((unsigned int *)v5, v42, v45 | v46 & 0xFFFFFFFFFFFFFFC0LL, v47);
      goto LABEL_34;
    }
    *(_DWORD *)(*(_QWORD *)(v5 + 8) + 16LL * v42 + 8) = v44;
    if ( v42 )
      break;
LABEL_34:
    if ( v67 == ++v43 )
      goto LABEL_41;
LABEL_35:
    sub_F03D40((__int64 *)v4, v42);
  }
  ++v43;
  v48 = *(_QWORD *)(v5 + 8) + 16LL * (v42 - 1);
  v49 = (unsigned __int64 *)(*(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 12));
  *v49 = v45 | *v49 & 0xFFFFFFFFFFFFFFC0LL;
  sub_325DE80(v5, v42, v47);
  if ( v67 != v43 )
    goto LABEL_35;
LABEL_41:
  if ( v67 - 1 != (_DWORD)v58 )
  {
    v70 = v5;
    v50 = v67 - 1;
    do
    {
      --v50;
      sub_F03AD0(v4, v42);
    }
    while ( v50 != (_DWORD)v58 );
    v5 = v70;
  }
  *(_DWORD *)(*(_QWORD *)(v5 + 8) + 16LL * v42 + 12) = HIDWORD(v58);
  v51 = *(_QWORD *)(v5 + 8) + 16LL * *(unsigned int *)(v5 + 16) - 16;
  v52 = *(_DWORD *)(v51 + 8);
  v13 = *(_DWORD *)(v51 + 12) == v52;
  result = sub_325E8E0(*(_QWORD *)v51, (unsigned int *)(v51 + 12), v52, v57, a3);
LABEL_6:
  v14 = *(_DWORD *)(v5 + 16);
  *(_DWORD *)(*(_QWORD *)(v5 + 8) + 16LL * (unsigned int)(v14 - 1) + 8) = result;
  if ( v14 != 1 )
  {
    v15 = *(_QWORD *)(v5 + 8) + 16LL * (unsigned int)(v14 - 2);
    v16 = (unsigned __int64 *)(*(_QWORD *)v15 + 8LL * *(unsigned int *)(v15 + 12));
    result = *v16 & 0xFFFFFFFFFFFFFFC0LL | (unsigned int)(result - 1);
    *v16 = result;
  }
  if ( v13 )
  {
    v17 = *(_DWORD *)(v5 + 16) - 1;
    if ( *(_DWORD *)(v5 + 16) != 1 )
      return sub_325DE80(v5, v17, a3);
  }
  return result;
}
