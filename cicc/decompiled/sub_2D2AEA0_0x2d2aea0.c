// Function: sub_2D2AEA0
// Address: 0x2d2aea0
//
__int64 __fastcall sub_2D2AEA0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  __int64 v9; // rsi
  unsigned int v10; // r13d
  __int64 v11; // r9
  __int64 v12; // rsi
  int v13; // r8d
  _QWORD *v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 *v20; // rsi
  __int64 result; // rax
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rcx
  _QWORD *v25; // rax
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rdx

  v8 = *(_QWORD *)a1;
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
      v25 = *(_QWORD **)(v8 + 200);
      *v14 = *v25;
      *v25 = v14;
      sub_2D2AEA0(a1, v10);
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
          *((_DWORD *)v14 + v17 + 32) = *((_DWORD *)v14 + v16 + 32);
        }
        while ( v13 != v15 );
        v12 = v11 + *(_QWORD *)(a1 + 8);
        v15 = *(_DWORD *)(v12 + 8);
      }
      *(_DWORD *)(v12 + 8) = v15 - 1;
      v18 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
      v19 = v15 - 2;
      v20 = (unsigned __int64 *)(*(_QWORD *)v18 + 8LL * *(unsigned int *)(v18 + 12));
      *v20 = v19 | *v20 & 0xFFFFFFFFFFFFFFC0LL;
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v11 + 12) == v15 - 1 )
      {
        sub_2D22A70(a1, v10, *((_DWORD *)v14 + v19 + 32));
        sub_F03D40((__int64 *)(a1 + 8), v10);
      }
    }
  }
  else
  {
    v26 = *(unsigned int *)(v8 + 196);
    v27 = *(_DWORD *)(v9 + 12) + 1;
    if ( (_DWORD)v26 != v27 )
    {
      do
      {
        v28 = v27;
        v29 = v27++ - 1;
        *(_QWORD *)(v8 + 8 * v29 + 8) = *(_QWORD *)(v8 + 8 * v28 + 8);
        *(_DWORD *)(v8 + 4 * v29 + 128) = *(_DWORD *)(v8 + 4 * v28 + 128);
      }
      while ( (_DWORD)v26 != v27 );
      v27 = *(_DWORD *)(v8 + 196);
    }
    v30 = v27 - 1;
    *(_DWORD *)(v8 + 196) = v30;
    v31 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)(v31 + 8) = v30;
    if ( !v30 )
    {
      *(_QWORD *)v8 = 0;
      *(_DWORD *)(v8 + 192) = 0;
      *(_QWORD *)(v8 + 184) = 0;
      memset(
        (void *)((v8 + 8) & 0xFFFFFFFFFFFFFFF8LL),
        0,
        8LL * (((unsigned int)v8 - (((_DWORD)v8 + 8) & 0xFFFFFFF8) + 192) >> 3));
      return sub_2D29C80(a1, 0, v31, 0, v26, a6);
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
