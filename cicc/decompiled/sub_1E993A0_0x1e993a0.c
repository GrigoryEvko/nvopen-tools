// Function: sub_1E993A0
// Address: 0x1e993a0
//
__int64 __fastcall sub_1E993A0(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 *v7; // rax
  char v8; // dl
  unsigned int v9; // r14d
  int v10; // eax
  int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 *v15; // rsi
  unsigned int v16; // edi
  __int64 *v17; // rcx
  _DWORD *v18; // rax
  int v19; // esi
  __int64 v20; // rax
  int v22; // [rsp+Ch] [rbp-34h]

  v22 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
  v7 = *(__int64 **)(a4 + 8);
  if ( *(__int64 **)(a4 + 16) != v7 )
  {
LABEL_2:
    sub_16CCBA0(a4, a2);
    if ( !v8 )
      return 1;
    goto LABEL_3;
  }
  v15 = &v7[*(unsigned int *)(a4 + 28)];
  v16 = *(_DWORD *)(a4 + 28);
  if ( v7 == v15 )
    goto LABEL_30;
  v17 = 0;
  do
  {
    if ( a2 == *v7 )
      return 1;
    if ( *v7 == -2 )
      v17 = v7;
    ++v7;
  }
  while ( v15 != v7 );
  if ( !v17 )
  {
LABEL_30:
    if ( v16 >= *(_DWORD *)(a4 + 24) )
      goto LABEL_2;
    *(_DWORD *)(a4 + 28) = v16 + 1;
    *v15 = a2;
    ++*(_QWORD *)a4;
  }
  else
  {
    *v17 = a2;
    --*(_DWORD *)(a4 + 32);
    ++*(_QWORD *)a4;
  }
LABEL_3:
  if ( *(_DWORD *)(a4 + 28) - *(_DWORD *)(a4 + 32) == 16 )
    return 0;
  if ( *(_DWORD *)(a2 + 40) != 1 )
  {
    v9 = 1;
    while ( 1 )
    {
      v11 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v9 + 8);
      if ( v11 == v22 )
        goto LABEL_10;
      v12 = sub_1E69D00(*(_QWORD *)(a1 + 232), v11);
      v13 = v12;
      if ( !v12 )
        return 0;
      v10 = **(unsigned __int16 **)(v12 + 16);
      if ( (_WORD)v10 != 15 )
        goto LABEL_7;
      v18 = *(_DWORD **)(v13 + 32);
      if ( (*v18 & 0xFFF00) == 0 && (v18[10] & 0xFFF00) == 0 )
      {
        v19 = v18[12];
        if ( v19 < 0 )
          break;
      }
LABEL_23:
      if ( *a3 )
        return 0;
      *a3 = v11;
LABEL_10:
      v9 += 2;
      if ( *(_DWORD *)(a2 + 40) == v9 )
        return 1;
    }
    v20 = sub_1E69D00(*(_QWORD *)(a1 + 232), v19);
    v13 = v20;
    if ( !v20 )
      return 0;
    v10 = **(unsigned __int16 **)(v20 + 16);
LABEL_7:
    if ( !v10 || v10 == 45 )
    {
      if ( !(unsigned __int8)sub_1E993A0(a1, v13, a3, a4) )
        return 0;
      goto LABEL_10;
    }
    goto LABEL_23;
  }
  return 1;
}
