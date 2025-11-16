// Function: sub_151ABA0
// Address: 0x151aba0
//
__int64 *__fastcall sub_151ABA0(__int64 *a1, __int64 a2, __int64 **a3)
{
  unsigned __int64 v4; // rax
  __int64 *v5; // r8
  unsigned __int64 v6; // rax
  __int64 v8; // r13
  __int64 v9; // r15
  _BYTE *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  unsigned int v14; // esi
  int v15; // r15d
  __int64 v16; // r9
  unsigned int v17; // r8d
  _DWORD *v18; // rdi
  int v19; // edx
  int v21; // ecx
  int v22; // ecx
  __int64 v23; // r10
  __int64 v24; // rsi
  int v25; // edx
  _DWORD *v26; // rax
  int v27; // edi
  int v28; // edi
  int v29; // esi
  int v30; // esi
  __int64 v31; // r9
  _DWORD *v32; // r8
  int v33; // r10d
  __int64 v34; // rcx
  int v35; // edi
  int v36; // r9d
  __int64 *v37; // [rsp+10h] [rbp-80h]
  int v38; // [rsp+10h] [rbp-80h]
  unsigned int v39; // [rsp+10h] [rbp-80h]
  _BYTE *v40; // [rsp+20h] [rbp-70h] BYREF
  __int64 v41; // [rsp+28h] [rbp-68h]
  _BYTE v42[16]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v43[2]; // [rsp+40h] [rbp-50h] BYREF
  char v44; // [rsp+50h] [rbp-40h]
  char v45; // [rsp+51h] [rbp-3Fh]

  v4 = *((unsigned int *)a3 + 2);
  if ( v4 <= 1 )
  {
    v45 = 1;
    v43[0] = "Invalid record";
    v44 = 3;
    sub_1514BE0(a1, (__int64)v43);
    return a1;
  }
  v5 = *a3;
  v6 = 8 * v4;
  v8 = **a3;
  v9 = (__int64)(v6 - 8) >> 3;
  v41 = 0x800000000LL;
  v10 = v42;
  v40 = v42;
  if ( v6 > 0x48 )
  {
    v37 = v5;
    sub_16CD150(&v40, v42, (__int64)(v6 - 8) >> 3, 1);
    v5 = v37;
    v10 = &v40[(unsigned int)v41];
  }
  v11 = 0;
  do
  {
    v10[v11] = v5[v11 + 1];
    ++v11;
  }
  while ( v9 != v11 );
  v12 = *(_QWORD *)(a2 + 248);
  LODWORD(v41) = v41 + v9;
  v13 = sub_1632050(v12, v40, (unsigned int)v41);
  v14 = *(_DWORD *)(a2 + 1000);
  v15 = v13;
  if ( !v14 )
  {
    ++*(_QWORD *)(a2 + 976);
    goto LABEL_14;
  }
  v16 = *(_QWORD *)(a2 + 984);
  v17 = (v14 - 1) & (37 * v8);
  v18 = (_DWORD *)(v16 + 8LL * v17);
  v19 = *v18;
  if ( (_DWORD)v8 != *v18 )
  {
    v38 = 1;
    v26 = 0;
    while ( v19 != -1 )
    {
      if ( v26 || v19 != -2 )
        v18 = v26;
      v17 = (v14 - 1) & (v38 + v17);
      v19 = *(_DWORD *)(v16 + 8LL * v17);
      if ( (_DWORD)v8 == v19 )
        goto LABEL_8;
      ++v38;
      v26 = v18;
      v18 = (_DWORD *)(v16 + 8LL * v17);
    }
    if ( !v26 )
      v26 = v18;
    v28 = *(_DWORD *)(a2 + 992);
    ++*(_QWORD *)(a2 + 976);
    v25 = v28 + 1;
    if ( 4 * (v28 + 1) < 3 * v14 )
    {
      if ( v14 - *(_DWORD *)(a2 + 996) - v25 > v14 >> 3 )
        goto LABEL_16;
      v39 = 37 * v8;
      sub_1392B70(a2 + 976, v14);
      v29 = *(_DWORD *)(a2 + 1000);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a2 + 984);
        v32 = 0;
        v33 = 1;
        LODWORD(v34) = v30 & v39;
        v25 = *(_DWORD *)(a2 + 992) + 1;
        v26 = (_DWORD *)(v31 + 8LL * (v30 & v39));
        v35 = *v26;
        if ( (_DWORD)v8 != *v26 )
        {
          while ( v35 != -1 )
          {
            if ( !v32 && v35 == -2 )
              v32 = v26;
            v34 = v30 & (unsigned int)(v34 + v33);
            v26 = (_DWORD *)(v31 + 8 * v34);
            v35 = *v26;
            if ( (_DWORD)v8 == *v26 )
              goto LABEL_16;
            ++v33;
          }
LABEL_28:
          if ( v32 )
            v26 = v32;
          goto LABEL_16;
        }
        goto LABEL_16;
      }
      goto LABEL_48;
    }
LABEL_14:
    sub_1392B70(a2 + 976, 2 * v14);
    v21 = *(_DWORD *)(a2 + 1000);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a2 + 984);
      LODWORD(v24) = v22 & (37 * v8);
      v25 = *(_DWORD *)(a2 + 992) + 1;
      v26 = (_DWORD *)(v23 + 8LL * (unsigned int)v24);
      v27 = *v26;
      if ( (_DWORD)v8 != *v26 )
      {
        v36 = 1;
        v32 = 0;
        while ( v27 != -1 )
        {
          if ( v27 == -2 && !v32 )
            v32 = v26;
          v24 = v22 & (unsigned int)(v24 + v36);
          v26 = (_DWORD *)(v23 + 8 * v24);
          v27 = *v26;
          if ( (_DWORD)v8 == *v26 )
            goto LABEL_16;
          ++v36;
        }
        goto LABEL_28;
      }
LABEL_16:
      *(_DWORD *)(a2 + 992) = v25;
      if ( *v26 != -1 )
        --*(_DWORD *)(a2 + 996);
      *v26 = v8;
      v26[1] = v15;
      *a1 = 1;
      goto LABEL_9;
    }
LABEL_48:
    ++*(_DWORD *)(a2 + 992);
    BUG();
  }
LABEL_8:
  v45 = 1;
  v43[0] = "Conflicting METADATA_KIND records";
  v44 = 3;
  sub_1514BE0(a1, (__int64)v43);
LABEL_9:
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  return a1;
}
