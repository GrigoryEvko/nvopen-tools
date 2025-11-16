// Function: sub_35988E0
// Address: 0x35988e0
//
__int64 __fastcall sub_35988E0(int a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // r12
  unsigned __int64 v18; // r15
  _QWORD *v19; // rcx
  _QWORD *v20; // rsi

  if ( a1 < 0 )
    v9 = *(_QWORD *)(*(_QWORD *)(a4 + 56) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v9 = *(_QWORD *)(*(_QWORD *)(a4 + 304) + 8LL * (unsigned int)a1);
  if ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
      goto LABEL_5;
    while ( 1 )
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        break;
      if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
      {
LABEL_5:
        if ( !v9 )
          break;
        while ( 1 )
        {
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
            if ( a3 == *(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) )
            {
              v9 = v10;
              goto LABEL_5;
            }
          }
          else if ( a3 == *(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) )
          {
            goto LABEL_12;
          }
          sub_2EAB0C0(v9, a2);
          v9 = v10;
          if ( !v10 )
            goto LABEL_12;
        }
      }
    }
  }
LABEL_12:
  v11 = *(unsigned int *)(a5 + 160);
  result = a2 & 0x7FFFFFFF;
  v13 = 8LL * (unsigned int)result;
  if ( (unsigned int)result >= (unsigned int)v11 || !*(_QWORD *)(*(_QWORD *)(a5 + 152) + 8LL * (unsigned int)result) )
  {
    v14 = result + 1;
    if ( (unsigned int)v11 < v14 && v14 != v11 )
    {
      if ( v14 >= v11 )
      {
        v17 = *(_QWORD *)(a5 + 168);
        v18 = v14 - v11;
        if ( v14 > (unsigned __int64)*(unsigned int *)(a5 + 164) )
        {
          sub_C8D5F0(a5 + 152, (const void *)(a5 + 168), v14, 8u, v14, a6);
          v11 = *(unsigned int *)(a5 + 160);
        }
        v15 = *(_QWORD *)(a5 + 152);
        v19 = (_QWORD *)(v15 + 8 * v11);
        v20 = &v19[v18];
        if ( v19 != v20 )
        {
          do
            *v19++ = v17;
          while ( v20 != v19 );
          LODWORD(v11) = *(_DWORD *)(a5 + 160);
          v15 = *(_QWORD *)(a5 + 152);
        }
        *(_DWORD *)(a5 + 160) = v18 + v11;
        goto LABEL_20;
      }
      *(_DWORD *)(a5 + 160) = v14;
    }
    v15 = *(_QWORD *)(a5 + 152);
LABEL_20:
    v16 = (__int64 *)(v15 + v13);
    result = sub_2E10F30(a2);
    *v16 = result;
  }
  return result;
}
