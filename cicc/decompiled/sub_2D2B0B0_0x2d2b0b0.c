// Function: sub_2D2B0B0
// Address: 0x2d2b0b0
//
__int64 __fastcall sub_2D2B0B0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 result; // rax
  unsigned int v12; // edx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r8
  __int64 v17; // rsi
  unsigned int **v18; // rcx
  unsigned __int64 *v19; // rsi
  unsigned int v20; // esi
  __int64 *v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // edx

  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v7 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v10 = *(unsigned int *)(v9 + 8);
  result = *(_QWORD *)v9;
  if ( (_DWORD)v10 == 1 )
  {
    v21 = *(__int64 **)(v8 + 200);
    v22 = *v21;
    *(_QWORD *)result = *v21;
    *v21 = result;
    result = sub_2D2AEA0(a1, *(_DWORD *)(v8 + 192), (__int64)v21, v22, v10, a6);
    if ( a2 )
    {
      if ( *(_DWORD *)(v8 + 192) )
      {
        v23 = *(_DWORD *)(a1 + 16);
        if ( v23 )
        {
          v24 = *(_QWORD *)(a1 + 8);
          v25 = *(_DWORD *)(v24 + 12);
          if ( v25 < *(_DWORD *)(v24 + 8) )
          {
            result = v24 + 28;
            while ( !v25 )
            {
              if ( result == v24 + 28 + 16LL * (v23 - 1) )
              {
                result = **(unsigned int **)(v24 + 16LL * v23 - 16);
                *(_DWORD *)v8 = result;
                return result;
              }
              v25 = *(_DWORD *)result;
              result += 16;
            }
          }
        }
      }
    }
  }
  else
  {
    v12 = *(_DWORD *)(v9 + 12) + 1;
    if ( (_DWORD)v10 != v12 )
    {
      do
      {
        v13 = v12;
        v14 = v12++ - 1;
        *(_DWORD *)(result + 8 * v14) = *(_DWORD *)(result + 8 * v13);
        *(_DWORD *)(result + 8 * v14 + 4) = *(_DWORD *)(result + 8 * v13 + 4);
        *(_DWORD *)(result + 4 * v14 + 128) = *(_DWORD *)(result + 4 * v13 + 128);
      }
      while ( (_DWORD)v10 != v12 );
      v7 = *(_QWORD *)(a1 + 8);
      v12 = *(_DWORD *)(v7 + 16LL * *(unsigned int *)(a1 + 16) - 8);
    }
    v15 = *(unsigned int *)(v8 + 192);
    *(_DWORD *)(v7 + 16 * v15 + 8) = v12 - 1;
    if ( (_DWORD)v15 )
    {
      v19 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v15 - 1))
                               + 8LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v15 - 1) + 12));
      *v19 = (v12 - 2) | *v19 & 0xFFFFFFFFFFFFFFC0LL;
    }
    v16 = *(_QWORD *)(a1 + 8);
    v17 = *(unsigned int *)(a1 + 16);
    v18 = (unsigned int **)(v16 + 16 * v17 - 16);
    if ( *((_DWORD *)v18 + 3) == v12 - 1 )
    {
      v20 = *(_DWORD *)(v8 + 192);
      if ( v20 )
      {
        sub_2D22A70(a1, v20, *(_DWORD *)(result + 8LL * (v12 - 2) + 4));
        v20 = *(_DWORD *)(v8 + 192);
      }
      return (__int64)sub_F03D40((__int64 *)(a1 + 8), v20);
    }
    else if ( a2 )
    {
      if ( (_DWORD)v17 )
      {
        result = v16 + 12;
        while ( !*(_DWORD *)result )
        {
          result += 16;
          if ( v16 + 16LL * (unsigned int)(v17 - 1) + 28 == result )
            goto LABEL_26;
        }
      }
      else
      {
LABEL_26:
        result = **v18;
        *(_DWORD *)v8 = result;
      }
    }
  }
  return result;
}
