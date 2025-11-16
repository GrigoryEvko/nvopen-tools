// Function: sub_1B15D10
// Address: 0x1b15d10
//
__int64 *__fastcall sub_1B15D10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        int a5,
        __int64 a6,
        __int64 a7,
        char a8,
        __int64 a9)
{
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 *result; // rax
  __int64 v18; // rdx
  __int64 *v19; // r14
  __int64 v20; // rsi
  __int64 *v21; // r12
  __int64 *v22; // r9
  __int64 *v23; // rdi
  unsigned int v24; // r10d
  __int64 *v25; // rax
  __int64 *v26; // rcx

  v14 = a8;
  v15 = a9;
  *(_QWORD *)a1 = 6;
  *(_QWORD *)(a1 + 8) = 0;
  if ( a2 )
  {
    *(_QWORD *)(a1 + 16) = a2;
    if ( a2 != -8 && a2 != -16 )
    {
      sub_164C220(a1);
      v15 = a9;
      v14 = a8;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
  }
  v16 = a1 + 104;
  *(_BYTE *)(a1 + 56) = v14;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 104;
  *(_QWORD *)(a1 + 80) = a1 + 104;
  *(_QWORD *)(a1 + 88) = 8;
  *(_DWORD *)(a1 + 96) = 0;
  result = *(__int64 **)(v15 + 16);
  *(_QWORD *)(a1 + 24) = a3;
  *(_DWORD *)(a1 + 32) = a4;
  *(_DWORD *)(a1 + 36) = a5;
  *(_QWORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 48) = a7;
  if ( result == *(__int64 **)(v15 + 8) )
    v18 = *(unsigned int *)(v15 + 28);
  else
    v18 = *(unsigned int *)(v15 + 24);
  v19 = &result[v18];
  if ( result != v19 )
  {
    while ( 1 )
    {
      v20 = *result;
      v21 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v19 == ++result )
        return result;
    }
    if ( result != v19 )
    {
      v22 = (__int64 *)(a1 + 104);
LABEL_20:
      v23 = &v22[*(unsigned int *)(a1 + 92)];
      v24 = *(_DWORD *)(a1 + 92);
      if ( v23 == v22 )
      {
LABEL_28:
        if ( v24 < *(_DWORD *)(a1 + 88) )
        {
          *(_DWORD *)(a1 + 92) = v24 + 1;
          *v23 = v20;
          v22 = *(__int64 **)(a1 + 72);
          ++*(_QWORD *)(a1 + 64);
          v16 = *(_QWORD *)(a1 + 80);
          goto LABEL_14;
        }
        goto LABEL_13;
      }
      v25 = v22;
      v26 = 0;
      while ( *v25 != v20 )
      {
        if ( *v25 == -2 )
          v26 = v25;
        if ( v23 == ++v25 )
        {
          if ( v26 )
          {
            *v26 = v20;
            v16 = *(_QWORD *)(a1 + 80);
            --*(_DWORD *)(a1 + 96);
            v22 = *(__int64 **)(a1 + 72);
            ++*(_QWORD *)(a1 + 64);
            break;
          }
          goto LABEL_28;
        }
      }
LABEL_14:
      while ( 1 )
      {
        result = v21 + 1;
        if ( v21 + 1 == v19 )
          break;
        v20 = *result;
        for ( ++v21; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v21 = result )
        {
          if ( v19 == ++result )
            return result;
          v20 = *result;
        }
        if ( v21 == v19 )
          return result;
        if ( v22 == (__int64 *)v16 )
          goto LABEL_20;
LABEL_13:
        sub_16CCBA0(a1 + 64, v20);
        v16 = *(_QWORD *)(a1 + 80);
        v22 = *(__int64 **)(a1 + 72);
      }
    }
  }
  return result;
}
