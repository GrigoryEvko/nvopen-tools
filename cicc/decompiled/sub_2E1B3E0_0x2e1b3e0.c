// Function: sub_2E1B3E0
// Address: 0x2e1b3e0
//
__int64 __fastcall sub_2E1B3E0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 *v11; // rsi
  __int64 result; // rax
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 *v15; // rcx
  __int64 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // r9
  unsigned __int64 *v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rdx
  unsigned int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // edx

  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v7 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v10 = *(unsigned int *)(v9 + 8);
  v11 = *(__int64 **)v9;
  if ( (_DWORD)v10 == 1 )
  {
    v22 = *(__int64 **)(v8 + 200);
    v23 = *v22;
    *v11 = *v22;
    *v22 = (__int64)v11;
    result = sub_2E1B190(a1, *(_DWORD *)(v8 + 192), v23, a4, a5, v10);
    if ( a2 )
    {
      if ( *(_DWORD *)(v8 + 192) )
      {
        v24 = *(_DWORD *)(a1 + 16);
        if ( v24 )
        {
          v25 = *(_QWORD *)(a1 + 8);
          v26 = *(_DWORD *)(v25 + 12);
          if ( v26 < *(_DWORD *)(v25 + 8) )
          {
            result = v25 + 28;
            while ( !v26 )
            {
              if ( result == v25 + 28 + 16LL * (v24 - 1) )
              {
                result = **(_QWORD **)(v25 + 16LL * v24 - 16);
                *(_QWORD *)v8 = result;
                return result;
              }
              v26 = *(_DWORD *)result;
              result += 16;
            }
          }
        }
      }
    }
  }
  else
  {
    result = (unsigned int)(*(_DWORD *)(v9 + 12) + 1);
    if ( (_DWORD)v10 != (_DWORD)result )
    {
      do
      {
        v13 = (unsigned int)result;
        v14 = (unsigned int)(result - 1);
        LODWORD(result) = result + 1;
        v15 = &v11[2 * v13];
        v16 = &v11[2 * v14];
        *v16 = *v15;
        v16[1] = v15[1];
        v11[v14 + 16] = v11[v13 + 16];
      }
      while ( (_DWORD)v10 != (_DWORD)result );
      v7 = *(_QWORD *)(a1 + 8);
      result = *(unsigned int *)(v7 + 16LL * *(unsigned int *)(a1 + 16) - 8);
    }
    v17 = *(unsigned int *)(v8 + 192);
    *(_DWORD *)(v7 + 16 * v17 + 8) = result - 1;
    if ( (_DWORD)v17 )
    {
      v21 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v17 - 1))
                               + 8LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v17 - 1) + 12));
      *v21 = (unsigned int)(result - 2) | *v21 & 0xFFFFFFFFFFFFFFC0LL;
    }
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(unsigned int *)(a1 + 16);
    v20 = v18 + 16 * v19 - 16;
    if ( *(_DWORD *)(v20 + 12) == (_DWORD)result - 1 )
    {
      sub_2E1A5E0(a1, *(_DWORD *)(v8 + 192), v11[2 * (unsigned int)(result - 2) + 1]);
      return (__int64)sub_F03D40((__int64 *)(a1 + 8), *(_DWORD *)(v8 + 192));
    }
    else if ( a2 )
    {
      if ( (_DWORD)v19 )
      {
        result = v18 + 12;
        while ( !*(_DWORD *)result )
        {
          result += 16;
          if ( v18 + 16LL * (unsigned int)(v19 - 1) + 28 == result )
            goto LABEL_24;
        }
      }
      else
      {
LABEL_24:
        result = **(_QWORD **)v20;
        *(_QWORD *)v8 = result;
      }
    }
  }
  return result;
}
