// Function: sub_1EEF100
// Address: 0x1eef100
//
__int64 __fastcall sub_1EEF100(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 (*v7)(void); // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned int v11; // r14d
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi

  v2 = *(_QWORD *)(a2 + 40);
  a1[30] = v2;
  if ( !*(_BYTE *)(v2 + 16) )
    return 0;
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4FC450C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_17;
  }
  a1[29] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(
             *(_QWORD *)(v5 + 8),
             &unk_4FC450C);
  v7 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v8 = 0;
  if ( v7 != sub_1D00B00 )
    v8 = v7();
  a1[31] = v8;
  v9 = a1[30];
  v10 = *(unsigned int *)(v9 + 32);
  if ( !*(_DWORD *)(v9 + 32) )
    return 0;
  v11 = 0;
  v12 = 0;
  do
  {
    v13 = a1[29];
    v14 = v12 & 0x7FFFFFFF;
    if ( (unsigned int)v14 < *(_DWORD *)(v13 + 408) )
    {
      v15 = *(_QWORD *)(*(_QWORD *)(v13 + 400) + 8 * v14);
      if ( v15 )
      {
        if ( *(_QWORD *)(v15 + 104) )
          v11 |= sub_1EED480(a1, v15);
      }
    }
    ++v12;
  }
  while ( v10 != v12 );
  return v11;
}
