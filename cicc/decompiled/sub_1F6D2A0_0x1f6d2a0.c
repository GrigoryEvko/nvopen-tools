// Function: sub_1F6D2A0
// Address: 0x1f6d2a0
//
__int64 *__fastcall sub_1F6D2A0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 *result; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD *v12; // rdx
  int v13; // edx
  int v14; // r9d

  v3 = *(_QWORD **)(a1 + 600);
  if ( *(_QWORD **)(a1 + 608) == v3 )
  {
    v12 = &v3[*(unsigned int *)(a1 + 620)];
    if ( v3 == v12 )
    {
LABEL_17:
      v3 = v12;
    }
    else
    {
      while ( a2 != *v3 )
      {
        if ( v12 == ++v3 )
          goto LABEL_17;
      }
    }
  }
  else
  {
    v3 = sub_16CC9F0(a1 + 592, a2);
    if ( a2 == *v3 )
    {
      v10 = *(_QWORD *)(a1 + 608);
      if ( v10 == *(_QWORD *)(a1 + 600) )
        v11 = *(unsigned int *)(a1 + 620);
      else
        v11 = *(unsigned int *)(a1 + 616);
      v12 = (_QWORD *)(v10 + 8 * v11);
    }
    else
    {
      v4 = *(_QWORD *)(a1 + 608);
      if ( v4 != *(_QWORD *)(a1 + 600) )
        goto LABEL_4;
      v3 = (_QWORD *)(v4 + 8LL * *(unsigned int *)(a1 + 620));
      v12 = v3;
    }
  }
  if ( v12 != v3 )
  {
    *v3 = -2;
    ++*(_DWORD *)(a1 + 624);
  }
LABEL_4:
  result = (__int64 *)*(unsigned int *)(a1 + 584);
  if ( (_DWORD)result )
  {
    v6 = *(_QWORD *)(a1 + 568);
    v7 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_6:
      result = (__int64 *)(v6 + 16LL * (_QWORD)result);
      if ( v8 != result )
      {
        result = *(__int64 **)(a1 + 32);
        result[*((unsigned int *)v8 + 2)] = 0;
        *v8 = -16;
        --*(_DWORD *)(a1 + 576);
        ++*(_DWORD *)(a1 + 580);
      }
    }
    else
    {
      v13 = 1;
      while ( v9 != -8 )
      {
        v14 = v13 + 1;
        v7 = ((_DWORD)result - 1) & (v13 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_6;
        v13 = v14;
      }
    }
  }
  return result;
}
