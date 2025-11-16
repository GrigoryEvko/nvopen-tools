// Function: sub_2FB6C60
// Address: 0x2fb6c60
//
__int64 __fastcall sub_2FB6C60(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rdx
  __int64 v9; // rsi
  unsigned int v10; // r13d
  __int64 v11; // r9
  __int64 v12; // rsi
  int v13; // r8d
  _QWORD *v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 *v19; // rdx
  __int64 v20; // rsi
  __int64 result; // rax
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rcx
  _QWORD *v25; // rax
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned int v30; // eax
  _QWORD *v31; // rax

  v8 = *(_QWORD **)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = a2 - 1;
  if ( v10 )
  {
    v11 = 16LL * v10;
    v12 = v11 + v9;
    v13 = *(_DWORD *)(v12 + 8);
    v14 = *(_QWORD **)v12;
    if ( v13 == 1 )
    {
      v25 = (_QWORD *)v8[24];
      *v14 = *v25;
      *v25 = v14;
      sub_2FB6C60(a1, v10);
    }
    else
    {
      v15 = *(_DWORD *)(v12 + 12) + 1;
      if ( v13 != v15 )
      {
        do
        {
          v16 = v15;
          v17 = v15++ - 1;
          v14[v17] = v14[v16];
          v14[v17 + 12] = v14[v16 + 12];
        }
        while ( v13 != v15 );
        v12 = v11 + *(_QWORD *)(a1 + 8);
        v15 = *(_DWORD *)(v12 + 8);
      }
      *(_DWORD *)(v12 + 8) = v15 - 1;
      v18 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
      v19 = (unsigned __int64 *)(*(_QWORD *)v18 + 8LL * *(unsigned int *)(v18 + 12));
      v20 = v15 - 2;
      *v19 = v20 | *v19 & 0xFFFFFFFFFFFFFFC0LL;
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v11 + 12) == v15 - 1 )
      {
        sub_2FB6C00(a1, v10, v14[v20 + 12]);
        sub_F03D40((__int64 *)(a1 + 8), v10);
      }
    }
  }
  else
  {
    v26 = *((unsigned int *)v8 + 47);
    v27 = *(_DWORD *)(v9 + 12) + 1;
    if ( (_DWORD)v26 != v27 )
    {
      do
      {
        v28 = v27;
        v29 = v27++ - 1;
        v8[v29 + 1] = v8[v28 + 1];
        v8[v29 + 12] = v8[v28 + 12];
      }
      while ( (_DWORD)v26 != v27 );
      v27 = *((_DWORD *)v8 + 47);
    }
    v30 = v27 - 1;
    *((_DWORD *)v8 + 47) = v30;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v30;
    if ( !v30 )
    {
      *((_DWORD *)v8 + 46) = 0;
      memset(v8, 0, 0xB8u);
      v31 = v8 + 18;
      do
      {
        *v8 = 0;
        v8 += 2;
        *(v8 - 1) = 0;
      }
      while ( v31 != v8 );
      return sub_2FB39C0(a1, 0, (__int64)v8, 0, v26, a6);
    }
  }
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
    {
      v22 = 16LL * a2;
      v23 = result + v22;
      v24 = *(_QWORD *)(*(_QWORD *)(result + 16LL * v10) + 8LL * *(unsigned int *)(result + 16LL * v10 + 12));
      *(_QWORD *)v23 = v24 & 0xFFFFFFFFFFFFFFC0LL;
      *(_DWORD *)(v23 + 8) = (v24 & 0x3F) + 1;
      result = *(_QWORD *)(a1 + 8);
      *(_DWORD *)(result + v22 + 12) = 0;
    }
  }
  return result;
}
