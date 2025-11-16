// Function: sub_34A2FF0
// Address: 0x34a2ff0
//
__int64 __fastcall sub_34A2FF0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
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
  unsigned int v22; // r8d
  __int64 *v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // edx

  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v7 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v10 = *(unsigned int *)(v9 + 8);
  v11 = *(__int64 **)v9;
  if ( (_DWORD)v10 == 1 )
  {
    v23 = *(__int64 **)(v8 + 200);
    v24 = *v23;
    *v11 = *v23;
    *v23 = (__int64)v11;
    result = sub_34A2E00(a1, *(_DWORD *)(v8 + 192), v24, a4, a5, v10);
    if ( a2 )
    {
      if ( *(_DWORD *)(v8 + 192) )
      {
        v25 = *(_DWORD *)(a1 + 16);
        if ( v25 )
        {
          v26 = *(_QWORD *)(a1 + 8);
          v27 = *(_DWORD *)(v26 + 12);
          if ( v27 < *(_DWORD *)(v26 + 8) )
          {
            result = v26 + 28;
            while ( !v27 )
            {
              if ( result == v26 + 28 + 16LL * (v25 - 1) )
              {
                result = **(_QWORD **)(v26 + 16LL * v25 - 16);
                *(_QWORD *)v8 = result;
                return result;
              }
              v27 = *(_DWORD *)result;
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
        *((_BYTE *)v11 + v14 + 176) = *((_BYTE *)v11 + v13 + 176);
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
      v22 = *(_DWORD *)(v8 + 192);
      if ( v22 )
      {
        sub_349D820(a1, v22, v11[2 * (unsigned int)(result - 2) + 1]);
        v22 = *(_DWORD *)(v8 + 192);
      }
      return (__int64)sub_F03D40((__int64 *)(a1 + 8), v22);
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
            goto LABEL_26;
        }
      }
      else
      {
LABEL_26:
        result = **(_QWORD **)v20;
        *(_QWORD *)v8 = result;
      }
    }
  }
  return result;
}
