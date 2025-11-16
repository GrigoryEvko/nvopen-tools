// Function: sub_2E21680
// Address: 0x2e21680
//
__int64 __fastcall sub_2E21680(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  bool v6; // r8
  char v7; // r8
  _QWORD *v8; // rsi
  __int64 v9; // rcx
  unsigned int v10; // r15d
  __int16 *v11; // r12
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // edx

  result = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)result )
    return result;
  v5 = a3;
  v6 = sub_2E21220((__int64)a1, a2, a3);
  result = 3;
  if ( v6 )
    return result;
  v7 = sub_2E212B0(a1, a2, v5);
  result = 2;
  if ( v7 )
    return result;
  v8 = *(_QWORD **)(a2 + 104);
  v9 = *(_QWORD *)(*a1 + 8LL) + 24 * v5;
  v10 = *(_DWORD *)(v9 + 16) & 0xFFF;
  v11 = (__int16 *)(*(_QWORD *)(*a1 + 56LL) + 2LL * (*(_DWORD *)(v9 + 16) >> 12));
  if ( !v8 )
  {
    do
    {
      if ( !v11 )
        break;
      v19 = sub_2E21610((__int64)a1, a2, v10);
      if ( (unsigned int)sub_2E1AC90(v19, 1u, v20, v21, v22, v23) )
        return 1;
      v24 = *v11++;
      v10 += v24;
    }
    while ( (_WORD)v24 );
    return 0;
  }
  v12 = (_QWORD *)(*(_QWORD *)(*a1 + 64LL) + 16LL * *(unsigned __int16 *)(v9 + 20));
  if ( !v11 )
    return 0;
  while ( 1 )
  {
    if ( v8 )
    {
      while ( (v8[14] & *v12) == 0 && (v12[1] & v8[15]) == 0 )
      {
        v8 = (_QWORD *)v8[13];
        if ( !v8 )
          goto LABEL_12;
      }
      v13 = sub_2E21610((__int64)a1, (__int64)v8, v10);
      if ( (unsigned int)sub_2E1AC90(v13, 1u, v14, v15, v16, v17) )
        return 1;
    }
LABEL_12:
    v18 = *v11;
    v12 += 2;
    ++v11;
    if ( !(_WORD)v18 )
      return 0;
    v8 = *(_QWORD **)(a2 + 104);
    v10 += v18;
  }
}
