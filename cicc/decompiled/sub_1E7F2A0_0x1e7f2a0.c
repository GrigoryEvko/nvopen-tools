// Function: sub_1E7F2A0
// Address: 0x1e7f2a0
//
__int64 __fastcall sub_1E7F2A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 (*v3)(); // rdx
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r13
  unsigned int v15; // r12d
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  _DWORD *v18; // rax
  _DWORD *j; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 i; // rdx
  __int64 v24; // rax

  *(_QWORD *)(a1 + 232) = a2;
  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 40LL);
  v4 = 0;
  if ( v3 != sub_1D00B00 )
    v4 = ((__int64 (__fastcall *)(_QWORD))v3)(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 240) = v4;
  v5 = *(__int64 (**)())(*(_QWORD *)v2 + 112LL);
  v6 = 0;
  if ( v5 != sub_1D00B10 )
    v6 = ((__int64 (__fastcall *)(__int64))v5)(v2);
  *(_QWORD *)(a1 + 248) = v6;
  v7 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 256) = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 40LL);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4FC6A0C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_29;
  }
  *(_QWORD *)(a1 + 264) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(
                            *(_QWORD *)(v8 + 8),
                            &unk_4FC6A0C);
  sub_1F4B6B0(a1 + 272, v2);
  v12 = *(_QWORD *)(a1 + 232);
  v13 = *(unsigned int *)(a1 + 560);
  v14 = (__int64)(*(_QWORD *)(v12 + 104) - *(_QWORD *)(v12 + 96)) >> 3;
  LODWORD(v2) = v14;
  if ( (unsigned int)v14 >= v13 )
  {
    if ( (unsigned int)v14 > v13 )
    {
      if ( (unsigned int)v14 > (unsigned __int64)*(unsigned int *)(a1 + 564) )
      {
        sub_16CD150(a1 + 552, (const void *)(a1 + 568), (unsigned int)v14, 8, v10, v11);
        v13 = *(unsigned int *)(a1 + 560);
      }
      v21 = *(_QWORD *)(a1 + 552);
      v22 = v21 + 8 * v13;
      for ( i = v21 + 8LL * (unsigned int)v14; i != v22; v22 += 8 )
      {
        if ( v22 )
        {
          *(_BYTE *)(v22 + 4) = 0;
          *(_DWORD *)v22 = -1;
        }
      }
      v24 = *(_QWORD *)(a1 + 232);
      *(_DWORD *)(a1 + 560) = v14;
      v2 = (__int64)(*(_QWORD *)(v24 + 104) - *(_QWORD *)(v24 + 96)) >> 3;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 560) = v14;
    v2 = (__int64)(*(_QWORD *)(v12 + 104) - *(_QWORD *)(v12 + 96)) >> 3;
  }
  v15 = *(_DWORD *)(a1 + 320) * v2;
  v16 = *(unsigned int *)(a1 + 608);
  if ( v15 < v16 )
    goto LABEL_19;
  if ( v15 > v16 )
  {
    if ( v15 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
    {
      sub_16CD150(a1 + 600, (const void *)(a1 + 616), v15, 4, v10, v11);
      v16 = *(unsigned int *)(a1 + 608);
    }
    v17 = *(_QWORD *)(a1 + 600);
    v18 = (_DWORD *)(v17 + 4 * v16);
    for ( j = (_DWORD *)(v17 + 4LL * v15); j != v18; ++v18 )
    {
      if ( v18 )
        *v18 = 0;
    }
LABEL_19:
    *(_DWORD *)(a1 + 608) = v15;
  }
  return 0;
}
