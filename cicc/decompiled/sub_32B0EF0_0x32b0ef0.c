// Function: sub_32B0EF0
// Address: 0x32b0ef0
//
unsigned __int64 *__fastcall sub_32B0EF0(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdx
  int v6; // r8d
  _QWORD *v7; // rsi
  unsigned int v8; // ecx
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15; // rdx
  unsigned __int64 *result; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  unsigned __int64 *v19; // rdx
  unsigned int v20; // r8d
  _QWORD *v21; // rax
  unsigned int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // edx

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  v5 = v3 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v6 = *(_DWORD *)(v5 + 8);
  v7 = *(_QWORD **)v5;
  if ( v6 == 1 )
  {
    v21 = *(_QWORD **)(v4 + 144);
    *v7 = *v21;
    *v21 = v7;
    result = (unsigned __int64 *)sub_32B0CB0(a1, *(_DWORD *)(v4 + 136));
    if ( a2 )
    {
      if ( *(_DWORD *)(v4 + 136) )
      {
        v22 = *(_DWORD *)(a1 + 16);
        if ( v22 )
        {
          v23 = *(_QWORD *)(a1 + 8);
          v24 = *(_DWORD *)(v23 + 12);
          if ( v24 < *(_DWORD *)(v23 + 8) )
          {
            result = (unsigned __int64 *)(v23 + 28);
            while ( !v24 )
            {
              if ( result == (unsigned __int64 *)(v23 + 28 + 16LL * (v22 - 1)) )
              {
                result = **(unsigned __int64 ***)(v23 + 16LL * v22 - 16);
                *(_QWORD *)v4 = result;
                return result;
              }
              v24 = *(_DWORD *)result;
              result += 2;
            }
          }
        }
      }
    }
  }
  else
  {
    v8 = *(_DWORD *)(v5 + 12) + 1;
    if ( v6 != v8 )
    {
      do
      {
        v9 = v8;
        v10 = v8++ - 1;
        v11 = &v7[2 * v9];
        v12 = &v7[2 * v10];
        *v12 = *v11;
        v12[1] = v11[1];
      }
      while ( v6 != v8 );
      v3 = *(_QWORD *)(a1 + 8);
      v8 = *(_DWORD *)(v3 + 16LL * *(unsigned int *)(a1 + 16) - 8);
    }
    v13 = *(unsigned int *)(v4 + 136);
    *(_DWORD *)(v3 + 16 * v13 + 8) = v8 - 1;
    if ( (_DWORD)v13 )
    {
      v19 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v13 - 1))
                               + 8LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v13 - 1) + 12));
      *v19 = (v8 - 2) | *v19 & 0xFFFFFFFFFFFFFFC0LL;
    }
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(unsigned int *)(a1 + 16);
    result = (unsigned __int64 *)(16 * v15);
    v17 = v14 + 16 * v15 - 16;
    if ( *(_DWORD *)(v17 + 12) == v8 - 1 )
    {
      v20 = *(_DWORD *)(v4 + 136);
      if ( v20 )
      {
        sub_325DE80(a1, v20, v7[2 * v8 - 3]);
        v20 = *(_DWORD *)(v4 + 136);
      }
      return sub_F03D40((__int64 *)(a1 + 8), v20);
    }
    else if ( a2 )
    {
      if ( (_DWORD)v15 )
      {
        result = (unsigned __int64 *)(v14 + 12);
        v18 = v14 + 16LL * (unsigned int)(v15 - 1) + 28;
        while ( !*(_DWORD *)result )
        {
          result += 2;
          if ( (unsigned __int64 *)v18 == result )
            goto LABEL_26;
        }
      }
      else
      {
LABEL_26:
        result = **(unsigned __int64 ***)v17;
        *(_QWORD *)v4 = result;
      }
    }
  }
  return result;
}
