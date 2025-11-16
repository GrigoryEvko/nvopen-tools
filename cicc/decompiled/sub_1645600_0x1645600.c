// Function: sub_1645600
// Address: 0x1645600
//
__int64 __fastcall sub_1645600(_QWORD *a1, _QWORD *a2, __int64 a3, char a4)
{
  _QWORD *v4; // r8
  __int64 v7; // rbx
  __int64 v8; // r10
  int v9; // r15d
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned int v12; // r12d
  __int64 v13; // rdi
  int v14; // r12d
  int v15; // r12d
  _QWORD *v16; // rdi
  __int64 v17; // r13
  int v18; // eax
  int v19; // eax
  int v20; // edi
  __int64 *v21; // rcx
  unsigned int v22; // eax
  __int64 *v23; // rsi
  __int64 v24; // rdx
  int v25; // eax
  int v26; // r15d
  unsigned int v27; // eax
  __int64 v28; // r10
  int v29; // r11d
  unsigned int v30; // r9d
  __int64 *v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // r8
  unsigned int v36; // r12d
  int v37; // eax
  unsigned int v38; // eax
  __int64 *v39; // rcx
  __int64 v40; // rdx
  size_t v42; // rax
  int v43; // eax
  int v44; // edi
  int v45; // eax
  int v46; // r12d
  int v47; // r12d
  _QWORD *v48; // rdi
  __int64 v49; // r9
  int v50; // eax
  unsigned int v51; // eax
  __int64 v52; // rdx
  int v53; // edi
  _QWORD *v54; // [rsp+8h] [rbp-A8h]
  __int64 *v55; // [rsp+10h] [rbp-A0h]
  int v56; // [rsp+1Ch] [rbp-94h]
  unsigned int v57; // [rsp+20h] [rbp-90h]
  __int64 v58; // [rsp+28h] [rbp-88h]
  _QWORD *v59; // [rsp+28h] [rbp-88h]
  __int64 v60; // [rsp+28h] [rbp-88h]
  __int64 v61; // [rsp+28h] [rbp-88h]
  __int64 v62; // [rsp+28h] [rbp-88h]
  unsigned __int64 v63; // [rsp+38h] [rbp-78h] BYREF
  void *s1; // [rsp+40h] [rbp-70h]
  __int64 v65; // [rsp+48h] [rbp-68h]
  char v66[16]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v67; // [rsp+60h] [rbp-50h] BYREF
  __int64 v68; // [rsp+68h] [rbp-48h]
  char v69[64]; // [rsp+70h] [rbp-40h] BYREF

  v4 = a1;
  v7 = *a1;
  s1 = a2;
  v65 = a3;
  v8 = *(_QWORD *)(v7 + 2448);
  v9 = *(_DWORD *)(v7 + 2464);
  v66[0] = a4;
  v58 = v8;
  if ( v9 )
  {
    v26 = v9 - 1;
    v67 = sub_1644300(a2, (__int64)&a2[a3]);
    v27 = sub_1644190((__int64 *)&v67, v66);
    v28 = v58;
    v4 = a1;
    v29 = 1;
    v30 = v26 & v27;
    v31 = (__int64 *)(v58 + 8LL * (v26 & v27));
    v32 = *v31;
    if ( *v31 != -8 )
    {
      while ( 1 )
      {
        if ( v32 != -16 && ((*(_DWORD *)(v32 + 8) & 0x200) != 0) == v66[0] )
        {
          v33 = *(unsigned int *)(v32 + 12);
          if ( v33 == v65 )
          {
            v42 = 8 * v33;
            v56 = v29;
            v57 = v30;
            v61 = v28;
            if ( !v42 )
              break;
            v54 = v4;
            v55 = v31;
            v43 = memcmp(s1, *(const void **)(v32 + 16), v42);
            v31 = v55;
            v4 = v54;
            v28 = v61;
            v30 = v57;
            v29 = v56;
            if ( !v43 )
              break;
          }
        }
        v30 = v26 & (v29 + v30);
        v31 = (__int64 *)(v28 + 8LL * v30);
        v32 = *v31;
        if ( *v31 == -8 )
          goto LABEL_18;
        ++v29;
      }
      if ( v31 != (__int64 *)(*(_QWORD *)(v7 + 2448) + 8LL * *(unsigned int *)(v7 + 2464)) )
        return *v31;
    }
LABEL_18:
    v7 = *v4;
  }
  v59 = v4;
  v10 = sub_145CBF0((__int64 *)(v7 + 2272), 32, 16);
  *(_QWORD *)(v10 + 8) = 1037;
  *(_QWORD *)(v10 + 16) = 0;
  *(_QWORD *)v10 = v59;
  *(_QWORD *)(v10 + 24) = 0;
  sub_1643FB0(v10, a2, a3, a4);
  v11 = *v59;
  v12 = *(_DWORD *)(*v59 + 2464LL);
  if ( v12 )
  {
    v34 = *(_QWORD **)(v10 + 16);
    v35 = *(_QWORD *)(v11 + 2448);
    v36 = v12 - 1;
    v37 = *(_DWORD *)(v10 + 8) >> 9;
    v68 = *(unsigned int *)(v10 + 12);
    v60 = v35;
    v67 = (unsigned __int64)v34;
    v69[0] = v37 & 1;
    v63 = sub_1644300(v34, (__int64)&v34[v68]);
    v38 = v36 & sub_1644190((__int64 *)&v63, v69);
    v39 = (__int64 *)(v60 + 8LL * v38);
    v40 = *v39;
    if ( v10 == *v39 )
      return v10;
    v44 = 1;
    v23 = 0;
    while ( v40 != -8 )
    {
      if ( !v23 && v40 == -16 )
        v23 = v39;
      v38 = v36 & (v44 + v38);
      v39 = (__int64 *)(v60 + 8LL * v38);
      v40 = *v39;
      if ( *v39 == v10 )
        return v10;
      ++v44;
    }
    v45 = *(_DWORD *)(v11 + 2456);
    v12 = *(_DWORD *)(v11 + 2464);
    v13 = v11 + 2440;
    if ( !v23 )
      v23 = v39;
    ++*(_QWORD *)(v11 + 2440);
    v25 = v45 + 1;
    if ( 4 * v25 < 3 * v12 )
    {
      if ( v12 - (v25 + *(_DWORD *)(v11 + 2460)) > v12 >> 3 )
        goto LABEL_7;
      sub_16453F0(v13, v12);
      v46 = *(_DWORD *)(v11 + 2464);
      if ( v46 )
      {
        v47 = v46 - 1;
        v48 = *(_QWORD **)(v10 + 16);
        v49 = *(_QWORD *)(v11 + 2448);
        v50 = *(_DWORD *)(v10 + 8) >> 9;
        v68 = *(unsigned int *)(v10 + 12);
        v62 = v49;
        v67 = (unsigned __int64)v48;
        v69[0] = v50 & 1;
        v63 = sub_1644300(v48, (__int64)&v48[v68]);
        v51 = v47 & sub_1644190((__int64 *)&v63, v69);
        v23 = (__int64 *)(v62 + 8LL * v51);
        v52 = *v23;
        if ( *v23 == v10 )
          goto LABEL_6;
        v53 = 1;
        v21 = 0;
        while ( v52 != -8 )
        {
          if ( !v21 && v52 == -16 )
            v21 = v23;
          v51 = v47 & (v53 + v51);
          v23 = (__int64 *)(v62 + 8LL * v51);
          v52 = *v23;
          if ( *v23 == v10 )
            goto LABEL_6;
          ++v53;
        }
LABEL_34:
        v25 = *(_DWORD *)(v11 + 2456) + 1;
        if ( v21 )
          v23 = v21;
        goto LABEL_7;
      }
LABEL_53:
      ++*(_DWORD *)(v11 + 2456);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v11 + 2440);
    v13 = v11 + 2440;
  }
  sub_16453F0(v13, 2 * v12);
  v14 = *(_DWORD *)(v11 + 2464);
  if ( !v14 )
    goto LABEL_53;
  v15 = v14 - 1;
  v16 = *(_QWORD **)(v10 + 16);
  v17 = *(_QWORD *)(v11 + 2448);
  v18 = *(_DWORD *)(v10 + 8) >> 9;
  v68 = *(unsigned int *)(v10 + 12);
  v67 = (unsigned __int64)v16;
  v69[0] = v18 & 1;
  v63 = sub_1644300(v16, (__int64)&v16[v68]);
  v19 = sub_1644190((__int64 *)&v63, v69);
  v20 = 1;
  v21 = 0;
  v22 = v15 & v19;
  v23 = (__int64 *)(v17 + 8LL * v22);
  v24 = *v23;
  if ( *v23 != v10 )
  {
    while ( v24 != -8 )
    {
      if ( !v21 && v24 == -16 )
        v21 = v23;
      v22 = v15 & (v20 + v22);
      v23 = (__int64 *)(v17 + 8LL * v22);
      v24 = *v23;
      if ( *v23 == v10 )
        goto LABEL_6;
      ++v20;
    }
    goto LABEL_34;
  }
LABEL_6:
  v25 = *(_DWORD *)(v11 + 2456) + 1;
LABEL_7:
  *(_DWORD *)(v11 + 2456) = v25;
  if ( *v23 != -8 )
    --*(_DWORD *)(v11 + 2460);
  *v23 = v10;
  return v10;
}
