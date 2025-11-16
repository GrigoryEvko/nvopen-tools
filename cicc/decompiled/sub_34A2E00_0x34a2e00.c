// Function: sub_34A2E00
// Address: 0x34a2e00
//
__int64 __fastcall sub_34A2E00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rsi
  __int64 v10; // rcx
  unsigned int v11; // r13d
  __int64 v12; // r9
  __int64 v13; // rcx
  int v14; // r8d
  _QWORD *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rcx
  unsigned __int64 *v21; // rdi
  __int64 result; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rdx

  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a2 - 1;
  if ( v11 )
  {
    v12 = 16LL * v11;
    v13 = v12 + v10;
    v14 = *(_DWORD *)(v13 + 8);
    v15 = *(_QWORD **)v13;
    if ( v14 == 1 )
    {
      v26 = *(_QWORD **)(v9 + 200);
      *v15 = *v26;
      *v26 = v15;
      sub_34A2E00(a1, v11);
    }
    else
    {
      v16 = *(_DWORD *)(v13 + 12) + 1;
      if ( v14 != v16 )
      {
        do
        {
          v17 = v16;
          v18 = v16++ - 1;
          v15[v18] = v15[v17];
          v15[v18 + 12] = v15[v17 + 12];
        }
        while ( v14 != v16 );
        v13 = v12 + *(_QWORD *)(a1 + 8);
        v16 = *(_DWORD *)(v13 + 8);
      }
      v19 = v16 - 2;
      *(_DWORD *)(v13 + 8) = v16 - 1;
      v20 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
      v21 = (unsigned __int64 *)(*(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 12));
      *v21 = v19 | *v21 & 0xFFFFFFFFFFFFFFC0LL;
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v12 + 12) == v16 - 1 )
      {
        sub_349D820(a1, v11, v15[v19 + 12]);
        sub_F03D40((__int64 *)(a1 + 8), v11);
      }
    }
  }
  else
  {
    v27 = *(unsigned int *)(v9 + 196);
    v28 = *(_DWORD *)(v10 + 12) + 1;
    if ( (_DWORD)v27 != v28 )
    {
      do
      {
        v29 = v28;
        v30 = v28++ - 1;
        *(_QWORD *)(v9 + 8 * v30 + 8) = *(_QWORD *)(v9 + 8 * v29 + 8);
        *(_QWORD *)(v9 + 8 * v30 + 96) = *(_QWORD *)(v9 + 8 * v29 + 96);
      }
      while ( (_DWORD)v27 != v28 );
      v28 = *(_DWORD *)(v9 + 196);
    }
    v31 = v28 - 1;
    *(_DWORD *)(v9 + 196) = v31;
    v32 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)(v32 + 8) = v31;
    if ( !v31 )
    {
      *(_DWORD *)(v9 + 192) = 0;
      memset((void *)v9, 0, 0xC0u);
      return sub_34A26E0(a1, 0, v32, 0, v27, a6);
    }
  }
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
    {
      v23 = 16LL * a2;
      v24 = result + v23;
      v25 = *(_QWORD *)(*(_QWORD *)(result + 16LL * v11) + 8LL * *(unsigned int *)(result + 16LL * v11 + 12));
      *(_QWORD *)v24 = v25 & 0xFFFFFFFFFFFFFFC0LL;
      *(_DWORD *)(v24 + 8) = (v25 & 0x3F) + 1;
      result = *(_QWORD *)(a1 + 8);
      *(_DWORD *)(result + v23 + 12) = 0;
    }
  }
  return result;
}
