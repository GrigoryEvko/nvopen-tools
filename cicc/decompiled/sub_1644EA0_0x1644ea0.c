// Function: sub_1644EA0
// Address: 0x1644ea0
//
__int64 __fastcall sub_1644EA0(__int64 *a1, _QWORD *a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rbx
  int v7; // r8d
  __int64 v8; // r10
  unsigned int v9; // r12d
  __int64 v10; // r9
  __int64 v11; // rdi
  int v12; // r12d
  __int64 v13; // r13
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rsi
  int v17; // r12d
  int v18; // eax
  int v19; // edi
  __int64 *v20; // rcx
  unsigned int v21; // eax
  __int64 *v22; // rsi
  __int64 v23; // rdx
  int v24; // eax
  int v25; // eax
  __int64 v26; // r10
  int v27; // r8d
  unsigned int v28; // r9d
  __int64 *v29; // rcx
  __int64 v30; // rax
  int i; // r11d
  __int64 **v32; // rdx
  unsigned __int64 *v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // r12d
  int v37; // eax
  unsigned int v38; // eax
  __int64 *v39; // rcx
  __int64 v40; // rdx
  __int64 v42; // rax
  size_t v43; // rax
  int v44; // eax
  int v45; // edi
  int v46; // eax
  int v47; // r12d
  unsigned __int64 *v48; // rax
  unsigned __int64 v49; // rdx
  __int64 v50; // rsi
  int v51; // r12d
  int v52; // eax
  unsigned int v53; // eax
  __int64 v54; // rdx
  int v55; // edi
  __int64 *v56; // [rsp+0h] [rbp-A0h]
  int v57; // [rsp+8h] [rbp-98h]
  unsigned int v58; // [rsp+Ch] [rbp-94h]
  __int64 v59; // [rsp+10h] [rbp-90h]
  __int64 v60; // [rsp+10h] [rbp-90h]
  int v61; // [rsp+10h] [rbp-90h]
  __int64 v62; // [rsp+10h] [rbp-90h]
  int v63; // [rsp+18h] [rbp-88h]
  __int64 v64; // [rsp+18h] [rbp-88h]
  __int64 v65; // [rsp+18h] [rbp-88h]
  __int64 v66; // [rsp+18h] [rbp-88h]
  unsigned __int64 v67; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v68; // [rsp+30h] [rbp-70h] BYREF
  void *s1; // [rsp+38h] [rbp-68h]
  __int64 v70; // [rsp+40h] [rbp-60h]
  char v71[8]; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v72; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v73; // [rsp+58h] [rbp-48h]
  __int64 v74; // [rsp+60h] [rbp-40h]
  char v75[56]; // [rsp+68h] [rbp-38h] BYREF

  v6 = *(_QWORD *)*a1;
  v68 = a1;
  s1 = a2;
  v7 = *(_DWORD *)(v6 + 2432);
  v8 = *(_QWORD *)(v6 + 2416);
  v70 = a3;
  v71[0] = a4;
  v59 = v8;
  v63 = v7;
  if ( v7 )
  {
    v72 = sub_1644300(a2, (__int64)&a2[a3]);
    v25 = sub_1644240(&v68, (__int64 *)&v72, v71);
    v26 = v59;
    v27 = v63 - 1;
    v28 = (v63 - 1) & v25;
    v29 = (__int64 *)(v59 + 8LL * v28);
    v30 = *v29;
    if ( *v29 != -8 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v30 != -16 )
        {
          v32 = *(__int64 ***)(v30 + 16);
          if ( v68 == *v32 && v71[0] == (*(_DWORD *)(v30 + 8) >> 8 != 0) )
          {
            v42 = (8LL * *(unsigned int *)(v30 + 12) - 8) >> 3;
            if ( v42 == v70 )
            {
              v43 = 8 * v42;
              v57 = i;
              v58 = v28;
              v61 = v27;
              v66 = v26;
              if ( !v43 )
                break;
              v56 = v29;
              v44 = memcmp(s1, v32 + 1, v43);
              v29 = v56;
              v26 = v66;
              v27 = v61;
              v28 = v58;
              i = v57;
              if ( !v44 )
                break;
            }
          }
        }
        v28 = v27 & (i + v28);
        v29 = (__int64 *)(v26 + 8LL * v28);
        v30 = *v29;
        if ( *v29 == -8 )
          goto LABEL_2;
      }
      if ( v29 != (__int64 *)(*(_QWORD *)(v6 + 2416) + 8LL * *(unsigned int *)(v6 + 2432)) )
        return *v29;
    }
  }
LABEL_2:
  v64 = sub_145CBF0((__int64 *)(v6 + 2272), 8 * a3 + 32, 8);
  sub_16433E0(v64, a1, a2, a3, a4);
  v9 = *(_DWORD *)(v6 + 2432);
  v10 = v64;
  if ( v9 )
  {
    v33 = *(unsigned __int64 **)(v64 + 16);
    v34 = *v33;
    v60 = *(_QWORD *)(v6 + 2416);
    v73 = v33 + 1;
    v72 = v34;
    v35 = *(unsigned int *)(v64 + 12);
    v74 = (v35 * 8 - 8) >> 3;
    v75[0] = *(_DWORD *)(v64 + 8) >> 8 != 0;
    v36 = v9 - 1;
    v67 = sub_1644300(v33 + 1, (__int64)&v33[v35]);
    v37 = sub_1644240(&v72, (__int64 *)&v67, v75);
    v10 = v64;
    v38 = v36 & v37;
    v39 = (__int64 *)(v60 + 8LL * v38);
    v40 = *v39;
    if ( v64 == *v39 )
      return v10;
    v45 = 1;
    v22 = 0;
    while ( v40 != -8 )
    {
      if ( v40 == -16 && !v22 )
        v22 = v39;
      v38 = v36 & (v45 + v38);
      v39 = (__int64 *)(v60 + 8LL * v38);
      v40 = *v39;
      if ( v64 == *v39 )
        return v10;
      ++v45;
    }
    v46 = *(_DWORD *)(v6 + 2424);
    v9 = *(_DWORD *)(v6 + 2432);
    v11 = v6 + 2408;
    if ( !v22 )
      v22 = v39;
    ++*(_QWORD *)(v6 + 2408);
    v24 = v46 + 1;
    if ( 4 * v24 < 3 * v9 )
    {
      if ( v9 - (v24 + *(_DWORD *)(v6 + 2428)) > v9 >> 3 )
        goto LABEL_7;
      sub_1644C70(v11, v9);
      v47 = *(_DWORD *)(v6 + 2432);
      if ( v47 )
      {
        v48 = *(unsigned __int64 **)(v64 + 16);
        v62 = *(_QWORD *)(v6 + 2416);
        v49 = *v48;
        v50 = *(unsigned int *)(v64 + 12);
        v73 = v48 + 1;
        v72 = v49;
        v74 = (v50 * 8 - 8) >> 3;
        v75[0] = *(_DWORD *)(v64 + 8) >> 8 != 0;
        v51 = v47 - 1;
        v67 = sub_1644300(v48 + 1, (__int64)&v48[v50]);
        v52 = sub_1644240(&v72, (__int64 *)&v67, v75);
        v10 = v64;
        v53 = v51 & v52;
        v22 = (__int64 *)(v62 + 8LL * v53);
        v54 = *v22;
        if ( v64 == *v22 )
          goto LABEL_6;
        v55 = 1;
        v20 = 0;
        while ( v54 != -8 )
        {
          if ( v54 == -16 && !v20 )
            v20 = v22;
          v53 = v51 & (v55 + v53);
          v22 = (__int64 *)(v62 + 8LL * v53);
          v54 = *v22;
          if ( v64 == *v22 )
            goto LABEL_6;
          ++v55;
        }
LABEL_34:
        v24 = *(_DWORD *)(v6 + 2424) + 1;
        if ( v20 )
          v22 = v20;
        goto LABEL_7;
      }
LABEL_53:
      ++*(_DWORD *)(v6 + 2424);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v6 + 2408);
    v11 = v6 + 2408;
  }
  v65 = v10;
  sub_1644C70(v11, 2 * v9);
  v12 = *(_DWORD *)(v6 + 2432);
  if ( !v12 )
    goto LABEL_53;
  v13 = *(_QWORD *)(v6 + 2416);
  v14 = *(unsigned __int64 **)(v65 + 16);
  v15 = *v14;
  v16 = *(unsigned int *)(v65 + 12);
  v73 = v14 + 1;
  v72 = v15;
  v74 = (v16 * 8 - 8) >> 3;
  v75[0] = *(_DWORD *)(v65 + 8) >> 8 != 0;
  v17 = v12 - 1;
  v67 = sub_1644300(v14 + 1, (__int64)&v14[v16]);
  v18 = sub_1644240(&v72, (__int64 *)&v67, v75);
  v10 = v65;
  v19 = 1;
  v20 = 0;
  v21 = v17 & v18;
  v22 = (__int64 *)(v13 + 8LL * v21);
  v23 = *v22;
  if ( v65 != *v22 )
  {
    while ( v23 != -8 )
    {
      if ( !v20 && v23 == -16 )
        v20 = v22;
      v21 = v17 & (v19 + v21);
      v22 = (__int64 *)(v13 + 8LL * v21);
      v23 = *v22;
      if ( v65 == *v22 )
        goto LABEL_6;
      ++v19;
    }
    goto LABEL_34;
  }
LABEL_6:
  v24 = *(_DWORD *)(v6 + 2424) + 1;
LABEL_7:
  *(_DWORD *)(v6 + 2424) = v24;
  if ( *v22 != -8 )
    --*(_DWORD *)(v6 + 2428);
  *v22 = v10;
  return v10;
}
