// Function: sub_1F4D060
// Address: 0x1f4d060
//
bool __fastcall sub_1F4D060(unsigned __int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v7; // r12
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rsi
  int v13; // edx
  unsigned int v14; // ecx
  unsigned int v15; // eax
  int v16; // edi
  unsigned int v17; // eax
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // r14
  __int64 v22; // r8
  __int64 v23; // r15
  __int64 i; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned int v27; // ecx
  __int64 *v28; // rax
  __int64 v29; // r8
  __int64 v30; // rbx
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rcx
  __int64 v34; // r15
  __int64 v35; // rdi
  _QWORD *v36; // rsi
  _QWORD *v37; // rdx
  int v38; // eax
  int v39; // r9d
  unsigned int v40; // [rsp+4h] [rbp-3Ch]

  v7 = a1;
  if ( !a3 )
    return (unsigned int)sub_1E165A0(v7, a2, 1, 0) != -1;
  if ( a2 >= 0 )
    return (unsigned int)sub_1E165A0(v7, a2, 1, 0) != -1;
  v10 = *(_QWORD *)(a3 + 272);
  v11 = *(_DWORD *)(v10 + 384);
  if ( !v11 )
    return (unsigned int)sub_1E165A0(v7, a2, 1, 0) != -1;
  v12 = *(_QWORD *)(v10 + 368);
  v13 = v11 - 1;
  v14 = (unsigned int)a1 >> 9;
  v15 = (unsigned int)a1 >> 4;
  v16 = 1;
  v17 = v13 & (v14 ^ v15);
  v18 = *(_QWORD *)(v12 + 16LL * v17);
  if ( v7 != v18 )
  {
    while ( v18 != -8 )
    {
      v17 = v13 & (v16 + v17);
      v18 = *(_QWORD *)(v12 + 16LL * v17);
      if ( v7 == v18 )
        goto LABEL_6;
      ++v16;
    }
    return (unsigned int)sub_1E165A0(v7, a2, 1, 0) != -1;
  }
LABEL_6:
  v19 = *(unsigned int *)(a3 + 408);
  v20 = a2 & 0x7FFFFFFF;
  v21 = a2 & 0x7FFFFFFF;
  v22 = 8 * v21;
  if ( (a2 & 0x7FFFFFFFu) >= (unsigned int)v19 || (v23 = *(_QWORD *)(*(_QWORD *)(a3 + 400) + 8LL * v20)) == 0 )
  {
    v32 = v20 + 1;
    if ( (unsigned int)v19 < v32 )
    {
      v34 = v32;
      if ( v32 < v19 )
      {
        *(_DWORD *)(a3 + 408) = v32;
        v33 = *(_QWORD *)(a3 + 400);
        goto LABEL_18;
      }
      if ( v32 > v19 )
      {
        if ( v32 > (unsigned __int64)*(unsigned int *)(a3 + 412) )
        {
          v40 = v32;
          sub_16CD150(a3 + 400, (const void *)(a3 + 416), v32, 8, 8 * a2, a6);
          v19 = *(unsigned int *)(a3 + 408);
          v32 = v40;
          v22 = 8LL * (a2 & 0x7FFFFFFF);
        }
        v33 = *(_QWORD *)(a3 + 400);
        v35 = *(_QWORD *)(a3 + 416);
        v36 = (_QWORD *)(v33 + 8 * v34);
        v37 = (_QWORD *)(v33 + 8 * v19);
        if ( v36 != v37 )
        {
          do
            *v37++ = v35;
          while ( v36 != v37 );
          v33 = *(_QWORD *)(a3 + 400);
        }
        *(_DWORD *)(a3 + 408) = v32;
        goto LABEL_18;
      }
    }
    v33 = *(_QWORD *)(a3 + 400);
LABEL_18:
    *(_QWORD *)(v33 + v22) = sub_1DBA290(a2);
    v23 = *(_QWORD *)(*(_QWORD *)(a3 + 400) + 8 * v21);
    sub_1DBB110((_QWORD *)a3, v23);
  }
  if ( !*(_DWORD *)(v23 + 72) )
    return 0;
  for ( i = *(_QWORD *)(a3 + 272); (*(_BYTE *)(v7 + 46) & 4) != 0; v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v25 = *(_QWORD *)(i + 368);
  v26 = *(unsigned int *)(i + 384);
  if ( (_DWORD)v26 )
  {
    v27 = (v26 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v28 = (__int64 *)(v25 + 16LL * v27);
    v29 = *v28;
    if ( *v28 == v7 )
      goto LABEL_14;
    v38 = 1;
    while ( v29 != -8 )
    {
      v39 = v38 + 1;
      v27 = (v26 - 1) & (v38 + v27);
      v28 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v7 )
        goto LABEL_14;
      v38 = v39;
    }
  }
  v28 = (__int64 *)(v25 + 16 * v26);
LABEL_14:
  v30 = v28[1];
  v31 = *(_QWORD *)(sub_1DB3C70((__int64 *)v23, v30) + 8);
  if ( (v31 & 6) == 0 )
    return 0;
  return (v31 & 0xFFFFFFFFFFFFFFF8LL) == (v30 & 0xFFFFFFFFFFFFFFF8LL);
}
