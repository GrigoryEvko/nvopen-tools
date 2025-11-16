// Function: sub_2DCC180
// Address: 0x2dcc180
//
__int64 __fastcall sub_2DCC180(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v6; // r12d
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r9
  __int64 v11; // r8
  unsigned int v12; // edi
  unsigned int *v13; // rax
  unsigned int v14; // ecx
  __int64 v15; // r8
  __int64 result; // rax
  unsigned __int64 v17; // r15
  __int64 i; // r15
  __int64 v19; // rdi
  unsigned __int64 v20; // rbx
  unsigned __int64 *v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int *v26; // rdx
  int v27; // eax
  int v28; // ecx
  int v29; // eax
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // eax
  unsigned int v33; // edi
  int v34; // r11d
  unsigned int *v35; // r10
  int v36; // eax
  int v37; // eax
  unsigned int *v38; // r8
  __int64 v39; // r15
  __int64 v40; // rdi
  int v41; // r10d
  unsigned int v42; // esi
  int v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]

  v6 = a2;
  v8 = sub_2FF6500(*(_QWORD *)(a1 + 24), a2, 1);
  v9 = *(_DWORD *)(a1 + 240);
  v10 = v8;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 216);
    goto LABEL_27;
  }
  v11 = *(_QWORD *)(a1 + 224);
  v12 = (v9 - 1) & (37 * v6);
  v13 = (unsigned int *)(v11 + 8LL * v12);
  v14 = *v13;
  if ( v6 == *v13 )
  {
LABEL_3:
    v15 = v13[1];
    goto LABEL_4;
  }
  v43 = 1;
  v26 = 0;
  while ( v14 != -1 )
  {
    if ( !v26 && v14 == -2 )
      v26 = v13;
    v12 = (v9 - 1) & (v43 + v12);
    v13 = (unsigned int *)(v11 + 8LL * v12);
    v14 = *v13;
    if ( v6 == *v13 )
      goto LABEL_3;
    ++v43;
  }
  if ( !v26 )
    v26 = v13;
  v27 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  v28 = v27 + 1;
  if ( 4 * (v27 + 1) >= 3 * v9 )
  {
LABEL_27:
    v44 = v10;
    sub_2DCBFB0(a1 + 216, 2 * v9);
    v29 = *(_DWORD *)(a1 + 240);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 224);
      v10 = v44;
      v32 = (v29 - 1) & (37 * v6);
      v28 = *(_DWORD *)(a1 + 232) + 1;
      v26 = (unsigned int *)(v31 + 8LL * v32);
      v33 = *v26;
      if ( v6 != *v26 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -1 )
        {
          if ( v33 == -2 && !v35 )
            v35 = v26;
          v32 = v30 & (v32 + v34);
          v26 = (unsigned int *)(v31 + 8LL * v32);
          v33 = *v26;
          if ( v6 == *v26 )
            goto LABEL_23;
          ++v34;
        }
        if ( v35 )
          v26 = v35;
      }
      goto LABEL_23;
    }
    goto LABEL_56;
  }
  if ( v9 - *(_DWORD *)(a1 + 236) - v28 <= v9 >> 3 )
  {
    v45 = v10;
    sub_2DCBFB0(a1 + 216, v9);
    v36 = *(_DWORD *)(a1 + 240);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = 0;
      v10 = v45;
      LODWORD(v39) = v37 & (37 * v6);
      v40 = *(_QWORD *)(a1 + 224);
      v41 = 1;
      v28 = *(_DWORD *)(a1 + 232) + 1;
      v26 = (unsigned int *)(v40 + 8LL * (unsigned int)v39);
      v42 = *v26;
      if ( v6 != *v26 )
      {
        while ( v42 != -1 )
        {
          if ( v42 == -2 && !v38 )
            v38 = v26;
          v39 = v37 & (unsigned int)(v39 + v41);
          v26 = (unsigned int *)(v40 + 8 * v39);
          v42 = *v26;
          if ( v6 == *v26 )
            goto LABEL_23;
          ++v41;
        }
        if ( v38 )
          v26 = v38;
      }
      goto LABEL_23;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 232);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a1 + 232) = v28;
  if ( *v26 != -1 )
    --*(_DWORD *)(a1 + 236);
  *v26 = v6;
  v15 = 0;
  v26[1] = 0;
LABEL_4:
  if ( a3 != (_QWORD *)(a4 + 48) )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 32) + 568LL))(
             *(_QWORD *)(a1 + 32),
             a4,
             a3,
             v6,
             v15,
             v10,
             *(_QWORD *)(a1 + 24),
             0,
             0);
  v17 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v17 )
    BUG();
  if ( (*(_QWORD *)v17 & 4) == 0 && (*(_BYTE *)(v17 + 44) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v17; ; i = *(_QWORD *)v17 )
    {
      v17 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v17 + 44) & 4) == 0 )
        break;
    }
  }
  (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 32) + 568LL))(
    *(_QWORD *)(a1 + 32),
    a4,
    v17,
    v6,
    v15,
    v10,
    *(_QWORD *)(a1 + 24),
    0,
    0);
  v19 = a4 + 40;
  if ( v17 == *(_QWORD *)(*(_QWORD *)(v17 + 24) + 56LL) )
  {
    sub_2E31080(v19, 0);
    BUG();
  }
  v20 = *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL;
  sub_2E31080(v19, v20);
  v21 = *(unsigned __int64 **)(v20 + 8);
  v22 = *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL;
  *v21 = v22 | *v21 & 7;
  *(_QWORD *)(v22 + 8) = v21;
  *(_QWORD *)(v20 + 8) = 0;
  *(_QWORD *)v20 &= 7uLL;
  if ( a3 == (_QWORD *)(*(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
    v23 = *(unsigned __int64 **)(a4 + 56);
  else
    v23 = *(unsigned __int64 **)(v17 + 8);
  sub_2E31040(a4 + 40, v20);
  v24 = *v23;
  v25 = *(_QWORD *)v20;
  *(_QWORD *)(v20 + 8) = v23;
  v24 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v20 = v24 | v25 & 7;
  *(_QWORD *)(v24 + 8) = v20;
  result = *v23 & 7;
  *v23 = result | v20;
  return result;
}
