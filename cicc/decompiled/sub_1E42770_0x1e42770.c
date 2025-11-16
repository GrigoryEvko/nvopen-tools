// Function: sub_1E42770
// Address: 0x1e42770
//
__int64 __fastcall sub_1E42770(int a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rdi
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // r12
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 *v16; // r12
  __int64 v17; // rsi
  _QWORD *v18; // rcx
  _QWORD *v19; // rdx

  if ( a1 < 0 )
    v9 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v9 = *(_QWORD *)(*(_QWORD *)(a4 + 272) + 8LL * (unsigned int)a1);
  if ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
      goto LABEL_5;
    do
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        goto LABEL_14;
    }
    while ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 );
    while ( 1 )
    {
LABEL_5:
      if ( !v9 )
        break;
      v10 = *(_QWORD *)(v9 + 32);
      if ( v10 )
      {
        do
        {
          if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
            break;
          v10 = *(_QWORD *)(v10 + 32);
        }
        while ( v10 );
        if ( *(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) != a3 )
          goto LABEL_20;
        v9 = v10;
      }
      else
      {
        if ( a3 == *(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) )
          break;
LABEL_20:
        sub_1E310D0(v9, a2);
        v9 = v10;
      }
    }
  }
LABEL_14:
  v11 = *(unsigned int *)(a5 + 408);
  result = a2 & 0x7FFFFFFF;
  v13 = 8LL * (unsigned int)result;
  if ( (unsigned int)result >= (unsigned int)v11 || !*(_QWORD *)(*(_QWORD *)(a5 + 400) + 8LL * (unsigned int)result) )
  {
    v14 = result + 1;
    if ( (unsigned int)v11 < (int)result + 1 )
    {
      if ( v14 < v11 )
      {
        *(_DWORD *)(a5 + 408) = v14;
      }
      else if ( v14 > v11 )
      {
        if ( v14 > (unsigned __int64)*(unsigned int *)(a5 + 412) )
        {
          sub_16CD150(a5 + 400, (const void *)(a5 + 416), v14, 8, a5, a6);
          v11 = *(unsigned int *)(a5 + 408);
        }
        v15 = *(_QWORD *)(a5 + 400);
        v17 = *(_QWORD *)(a5 + 416);
        v18 = (_QWORD *)(v15 + 8LL * v14);
        v19 = (_QWORD *)(v15 + 8 * v11);
        if ( v18 != v19 )
        {
          do
            *v19++ = v17;
          while ( v18 != v19 );
          v15 = *(_QWORD *)(a5 + 400);
        }
        *(_DWORD *)(a5 + 408) = v14;
        goto LABEL_18;
      }
    }
    v15 = *(_QWORD *)(a5 + 400);
LABEL_18:
    v16 = (__int64 *)(v15 + v13);
    result = sub_1DBA290(a2);
    *v16 = result;
  }
  return result;
}
