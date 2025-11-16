// Function: sub_ED5180
// Address: 0xed5180
//
__int64 __fastcall sub_ED5180(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // edx
  int v7; // eax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // rcx
  int v13; // edi
  unsigned __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx

  if ( !a4 )
    return a2;
  if ( !(_DWORD)a3 )
    return sub_ED50F0(a4, a2, a3, a4, a5, a6);
  if ( (_DWORD)a3 != 2 )
    return a2;
  v6 = *(_DWORD *)(a4 + 380);
  if ( !v6 )
    return 0;
  v7 = *(_DWORD *)(a4 + 376);
  v8 = *(_QWORD *)(a4 + 280);
  if ( v7 )
  {
    if ( a2 < v8 || a2 >= *(_QWORD *)(a4 + 8LL * (unsigned int)(v6 - 1) + 328) )
      return 0;
    if ( a2 < *(_QWORD *)(a4 + 328) )
    {
      v11 = 0;
    }
    else
    {
      v10 = 0;
      do
        v11 = ++v10;
      while ( a2 >= *(_QWORD *)(a4 + 8LL * v10 + 328) );
    }
    v12 = *(_QWORD *)(a4 + 8 * v11 + 288);
    v13 = v7 - 1;
    if ( v7 != 1 )
    {
      do
      {
        v14 = v12 & 0xFFFFFFFFFFFFFFC0LL;
        if ( a2 < *(_QWORD *)(v14 + 96) )
        {
          v16 = 0;
        }
        else
        {
          v15 = 0;
          do
            v16 = ++v15;
          while ( a2 >= *(_QWORD *)(v14 + 8LL * v15 + 96) );
        }
        v12 = *(_QWORD *)(v14 + 8 * v16);
        --v13;
      }
      while ( v13 );
    }
    v17 = v12 & 0xFFFFFFFFFFFFFFC0LL;
    if ( a2 < *(_QWORD *)(v17 + 8) )
    {
      v19 = 0;
    }
    else
    {
      v18 = v17 + 24;
      LODWORD(v19) = 0;
      do
      {
        v18 += 16LL;
        v19 = (unsigned int)(v19 + 1);
      }
      while ( a2 >= *(_QWORD *)(v18 - 16) );
    }
    if ( a2 < *(_QWORD *)(v17 + 16 * v19) )
      return 0;
    return *(_QWORD *)(v17 + 8 * v19 + 128);
  }
  else
  {
    if ( a2 < v8 || a2 >= *(_QWORD *)(a4 + 16LL * (unsigned int)(v6 - 1) + 288) )
      return 0;
    v9 = 0;
    if ( a2 >= *(_QWORD *)(a4 + 288) )
    {
      v9 = 1;
      if ( a2 >= *(_QWORD *)(a4 + 304) )
        v9 = (unsigned int)(a2 >= *(_QWORD *)(a4 + 320)) + 2;
      if ( a2 < *(_QWORD *)(a4 + 16 * v9 + 280) )
        return 0;
    }
    return *(_QWORD *)(a4 + 8 * v9 + 344);
  }
}
