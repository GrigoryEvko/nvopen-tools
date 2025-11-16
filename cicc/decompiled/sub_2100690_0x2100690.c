// Function: sub_2100690
// Address: 0x2100690
//
__int64 __fastcall sub_2100690(__int64 a1, __int64 a2, int *a3)
{
  unsigned __int64 v6; // rdx
  __int64 i; // rcx
  __int64 v8; // rdi
  __int64 v9; // rcx
  unsigned int v10; // esi
  __int64 *v11; // rax
  __int64 v12; // r9
  unsigned __int64 v13; // r12
  __int64 *v14; // rdx
  __int64 (*v15)(); // rax
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 v18; // rbx
  int v19; // r13d
  __int64 *v21; // rdx
  int v22; // eax
  int v23; // r10d

  v6 = *((_QWORD *)a3 + 2);
  for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
        (*(_BYTE *)(v6 + 46) & 4) != 0;
        v6 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL )
  {
    ;
  }
  v8 = *(_QWORD *)(i + 368);
  v9 = *(unsigned int *)(i + 384);
  if ( !(_DWORD)v9 )
  {
LABEL_18:
    v11 = (__int64 *)(v8 + 16 * v9);
    goto LABEL_5;
  }
  v10 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (__int64 *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v6 != *v11 )
  {
    v22 = 1;
    while ( v12 != -8 )
    {
      v23 = v22 + 1;
      v10 = (v9 - 1) & (v22 + v10);
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v6 == *v11 )
        goto LABEL_5;
      v22 = v23;
    }
    goto LABEL_18;
  }
LABEL_5:
  v13 = v11[1] & 0xFFFFFFFFFFFFFFF8LL;
  v14 = (__int64 *)sub_1DB3C70((__int64 *)a2, v13);
  if ( v14 == (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
    || (*(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v14 >> 1) & 3) > *(_DWORD *)(v13 + 24)
    || v13 != (v14[1] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v15 = *(__int64 (**)())(**(_QWORD **)(**(_QWORD **)(a1 + 24) + 16LL) + 112LL);
    if ( v15 == sub_1D00B10 )
      BUG();
    v16 = v15();
    v17 = *a3;
    v18 = *(_QWORD *)(a2 + 104);
    v19 = *(_DWORD *)(*(_QWORD *)(v16 + 248) + 4LL * ((v17 >> 8) & 0xFFF));
    if ( !v18 )
      return 0;
    while ( 1 )
    {
      if ( (*(_DWORD *)(v18 + 112) & v19) != 0 )
      {
        v21 = (__int64 *)sub_1DB3C70((__int64 *)v18, v13);
        if ( v21 != (__int64 *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8))
          && (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) <= *(_DWORD *)(v13 + 24)
          && v13 == (v21[1] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          break;
        }
      }
      v18 = *(_QWORD *)(v18 + 104);
      if ( !v18 )
        return 0;
    }
  }
  return 1;
}
