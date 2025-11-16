// Function: sub_2D349B0
// Address: 0x2d349b0
//
unsigned __int64 __fastcall sub_2D349B0(unsigned int *a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  unsigned int *v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rsi
  int v9; // r14d
  unsigned int *v10; // r13
  unsigned int v11; // r15d
  unsigned __int64 result; // rax
  bool v13; // r14
  unsigned int v14; // esi
  __int64 v15; // rdx
  unsigned __int64 *v16; // rcx
  int v17; // esi
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // rax
  unsigned __int64 v26; // r14
  __int64 v27; // rcx
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned int v30; // r9d
  unsigned int v31; // r14d
  unsigned int v32; // edx
  __int64 v33; // r15
  unsigned int v34; // eax
  int v35; // r14d
  __int64 v36; // rax
  unsigned int v37; // edx
  unsigned int v38; // r9d
  char v39; // si
  unsigned __int64 v40; // rax
  __int64 v41; // r15
  int v42; // esi
  bool v43; // zf
  unsigned int v44; // edx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rdx
  unsigned int v49; // r14d
  __int64 *v50; // r13
  __int64 v51; // rbx
  int v52; // r9d
  __int64 v53; // rdx
  __int64 v54; // rdi
  int v55; // ecx
  __int64 v56; // rax
  unsigned __int64 *v57; // rax
  unsigned int i; // r15d
  __int64 v59; // rax
  unsigned int v60; // edx
  unsigned int v61; // eax
  __int64 v62; // rsi
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // [rsp+8h] [rbp-A8h]
  unsigned int v66; // [rsp+14h] [rbp-9Ch]
  char v67; // [rsp+14h] [rbp-9Ch]
  __int64 v68; // [rsp+18h] [rbp-98h]
  __int64 v69; // [rsp+20h] [rbp-90h]
  unsigned int v70; // [rsp+28h] [rbp-88h]
  unsigned int v71; // [rsp+28h] [rbp-88h]
  unsigned int v72; // [rsp+2Ch] [rbp-84h]
  unsigned int v73; // [rsp+2Ch] [rbp-84h]
  unsigned __int64 v74; // [rsp+30h] [rbp-80h]
  unsigned __int64 v76; // [rsp+38h] [rbp-78h]
  unsigned int v77[4]; // [rsp+40h] [rbp-70h] BYREF
  _DWORD v78[4]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v79[10]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a1 + 2;
  v6 = a1[4];
  v74 = __PAIR64__(a3, a4);
  if ( !v6 || (v7 = *((_QWORD *)a1 + 1), *(_DWORD *)(v7 + 12) >= *(_DWORD *)(v7 + 8)) )
  {
    v22 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 192LL);
    sub_F03AD0(v5, *(_DWORD *)(*(_QWORD *)a1 + 192LL));
    ++*(_DWORD *)(*((_QWORD *)a1 + 1) + v22 + 12);
    v6 = a1[4];
    v7 = *((_QWORD *)a1 + 1);
  }
  v8 = v7 + 16LL * v6 - 16;
  v9 = *(_DWORD *)(v8 + 12);
  v10 = *(unsigned int **)v8;
  if ( !v9 && *v10 > a2 )
  {
    v23 = sub_F03A30((__int64 *)v5, v6 - 1);
    if ( v23 )
    {
      v24 = v23;
      v25 = v23 & 0x3F;
      v26 = v24 & 0xFFFFFFFFFFFFFFC0LL;
      v27 = a1[4];
      v8 = *((_QWORD *)a1 + 1) + 16 * v27 - 16;
      v10 = *(unsigned int **)v8;
      if ( *(_DWORD *)(v26 + 4 * v25 + 128) != (_DWORD)v74 || *(_DWORD *)(v26 + 8 * v25 + 4) != a2 )
      {
        v9 = *(_DWORD *)(v8 + 12);
        goto LABEL_5;
      }
      v76 = v25;
      sub_F03AD0(v5, v27 - 1);
      v21 = *v10;
      result = v76;
      if ( (unsigned int)v21 >= HIDWORD(v74) )
      {
        v18 = (unsigned int)v74;
        if ( __PAIR64__(v21, v10[32]) != v74 )
        {
          *(_DWORD *)(v26 + 8 * v76 + 4) = HIDWORD(v74);
          v17 = a1[4] - 1;
          if ( a1[4] != 1 )
            return sub_2D22A70((__int64)a1, v17, SHIDWORD(v74));
          return result;
        }
      }
      a2 = *(_DWORD *)(v26 + 8 * v76);
      sub_2D2B0B0((__int64)a1, 0, v21, v18, v19, v20);
    }
    else
    {
      **(_DWORD **)a1 = a2;
    }
    v8 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
    v9 = *(_DWORD *)(v8 + 12);
    v10 = *(unsigned int **)v8;
  }
LABEL_5:
  v11 = *(_DWORD *)(v8 + 8);
  result = sub_2D28A50((__int64)v10, (unsigned int *)(v8 + 12), v11, a2, SHIDWORD(v74), v74);
  v13 = v11 == v9;
  if ( (unsigned int)result <= 0x10 )
    goto LABEL_6;
  v28 = a1[4] - 1;
  v72 = *(_DWORD *)(*((_QWORD *)a1 + 1) + 16 * v28 + 12);
  v29 = sub_F03A30((__int64 *)v5, a1[4] - 1);
  v30 = v72;
  v69 = v29;
  if ( v29 )
  {
    v32 = 1;
    v73 = 2;
    v79[0] = v29 & 0xFFFFFFFFFFFFFFC0LL;
    v31 = (v29 & 0x3F) + 1;
    v77[0] = v31;
    v30 += v31;
  }
  else
  {
    v73 = 1;
    v31 = 0;
    v32 = 0;
  }
  v33 = *((_QWORD *)a1 + 1) + 16 * v28;
  v34 = *(_DWORD *)(v33 + 8);
  v66 = v30;
  v70 = v32;
  v77[v32] = v34;
  v35 = v34 + v31;
  v68 = v32;
  v79[v32] = *(_QWORD *)v33;
  v36 = sub_F03C90((__int64 *)v5, v28);
  v37 = v70;
  v38 = v66;
  if ( v36 )
  {
    v39 = v36;
    v40 = v36 & 0xFFFFFFFFFFFFFFC0LL;
    v41 = v73;
    v71 = v70 + 2;
    v42 = (v39 & 0x3F) + 1;
    v79[v73] = v40;
    v35 += v42;
    v43 = v37 == 0;
    v44 = 32;
    v77[v73] = v42;
    if ( !v43 )
      v44 = 48;
    if ( v35 + 1 > v44 )
      goto LABEL_26;
LABEL_48:
    v67 = 0;
    v73 = 0;
    goto LABEL_31;
  }
  v61 = 16;
  if ( v73 != 1 )
    v61 = 32;
  if ( v35 + 1 <= v61 )
  {
    v71 = v73;
    goto LABEL_48;
  }
  if ( v73 != 1 )
  {
    v73 = v70;
    v42 = v77[v68];
    v41 = v68;
    v71 = 2;
    v40 = v79[v68];
LABEL_26:
    v45 = v71++;
    goto LABEL_27;
  }
  v42 = v77[1];
  v40 = v79[1];
  v45 = 1;
  v71 = 2;
  v41 = 1;
LABEL_27:
  v79[v45] = v40;
  v46 = *(_QWORD *)a1;
  v77[v45] = v42;
  v77[v41] = 0;
  v47 = *(_QWORD *)(v46 + 200);
  v48 = *(_QWORD **)v47;
  if ( *(_QWORD *)v47 )
  {
    *(_QWORD *)v47 = *v48;
  }
  else
  {
    v62 = *(_QWORD *)(v47 + 8);
    *(_QWORD *)(v47 + 88) += 192LL;
    v63 = (v62 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_QWORD *)(v47 + 16) >= v63 + 192 && v62 )
    {
      *(_QWORD *)(v47 + 8) = v63 + 192;
      if ( !v63 )
        goto LABEL_30;
      v48 = (_QWORD *)((v62 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    }
    else
    {
      v64 = sub_9D1E70(v47 + 8, 192, 192, 6);
      v38 = v66;
      v48 = (_QWORD *)v64;
    }
  }
  *v48 = 0;
  v48[23] = 0;
  memset(
    (void *)((unsigned __int64)(v48 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v48 - (((_DWORD)v48 + 8) & 0xFFFFFFF8) + 192) >> 3));
LABEL_30:
  v79[v41] = v48;
  v67 = 1;
LABEL_31:
  v65 = sub_F03E60(v71, v35, 16, (__int64)v77, (__int64)v78, v38, 1u);
  sub_2D2D150((__int64)v79, v71, v77, (__int64)v78);
  if ( v69 )
    sub_F03AD0(v5, v28);
  v49 = v28;
  v50 = (__int64 *)(a1 + 2);
  v51 = 0;
  while ( 1 )
  {
    v52 = v78[v51];
    v53 = v79[v51];
    v54 = (unsigned int)(v52 - 1);
    v55 = *(_DWORD *)(v53 + 8 * v54 + 4);
    if ( v73 == (_DWORD)v51 && v67 )
    {
      v49 += (unsigned __int8)sub_2D34700(a1, v49, v54 | v53 & 0xFFFFFFFFFFFFFFC0LL, v55);
      goto LABEL_37;
    }
    *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * v49 + 8) = v52;
    if ( v49 )
      break;
LABEL_37:
    if ( v71 == ++v51 )
      goto LABEL_44;
LABEL_38:
    sub_F03D40(v50, v49);
  }
  ++v51;
  v56 = *((_QWORD *)a1 + 1) + 16LL * (v49 - 1);
  v57 = (unsigned __int64 *)(*(_QWORD *)v56 + 8LL * *(unsigned int *)(v56 + 12));
  *v57 = v54 | *v57 & 0xFFFFFFFFFFFFFFC0LL;
  sub_2D22A70((__int64)a1, v49, v55);
  if ( v71 != v51 )
    goto LABEL_38;
LABEL_44:
  for ( i = v71 - 1; i != (_DWORD)v65; --i )
    sub_F03AD0((unsigned int *)v50, v49);
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * v49 + 12) = HIDWORD(v65);
  v59 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
  v60 = *(_DWORD *)(v59 + 8);
  v13 = *(_DWORD *)(v59 + 12) == v60;
  result = sub_2D28A50(*(_QWORD *)v59, (unsigned int *)(v59 + 12), v60, a2, SHIDWORD(v74), v74);
LABEL_6:
  v14 = a1[4];
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * (v14 - 1) + 8) = result;
  if ( v14 != 1 )
  {
    v15 = *((_QWORD *)a1 + 1) + 16LL * (v14 - 2);
    v16 = (unsigned __int64 *)(*(_QWORD *)v15 + 8LL * *(unsigned int *)(v15 + 12));
    result = *v16 & 0xFFFFFFFFFFFFFFC0LL | (unsigned int)(result - 1);
    *v16 = result;
  }
  if ( v13 )
  {
    v17 = a1[4] - 1;
    if ( a1[4] != 1 )
      return sub_2D22A70((__int64)a1, v17, SHIDWORD(v74));
  }
  return result;
}
