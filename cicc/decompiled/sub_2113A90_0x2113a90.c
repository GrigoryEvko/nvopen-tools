// Function: sub_2113A90
// Address: 0x2113a90
//
__int64 __fastcall sub_2113A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned int v7; // r11d
  unsigned int v8; // edi
  __int64 v9; // rdx
  __int64 v10; // r9
  _DWORD *v11; // rax
  __int64 v12; // r9
  unsigned int v13; // ecx
  unsigned int v14; // edx
  int v15; // r8d
  unsigned int v16; // esi
  __int64 v17; // r13
  __int64 v18; // rbx
  void *v19; // rax
  void *v20; // rdi
  __int64 result; // rax
  __int64 v22; // rax

  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)a1 = &unk_4A01048;
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  if ( a2 && (v5 = *(_QWORD *)(a2 + 96)) != 0 )
  {
    v7 = 0;
    v8 = 1;
    while ( 1 )
    {
      v9 = v5 + 10LL * v7;
      v10 = *(unsigned __int16 *)(v9 + 4);
      if ( ((unsigned __int16)v10 & *(_WORD *)(v9 + 2)) == -1 )
        break;
      v11 = (_DWORD *)(*(_QWORD *)(a2 + 72) + 16LL * *(unsigned __int16 *)(v9 + 2));
      v12 = *(_QWORD *)(a2 + 72) + 16 * v10;
      if ( (_DWORD *)v12 != v11 )
      {
        v13 = 0;
        v14 = 0;
        do
        {
          v15 = v11[2];
          v16 = v14 + *v11;
          if ( v13 < v16 )
            v13 = v14 + *v11;
          v14 += v15;
          if ( v15 < 0 )
            v14 = v16;
          v11 += 4;
        }
        while ( v11 != (_DWORD *)v12 );
        if ( v13 > v8 )
        {
          do
            v8 *= 2;
          while ( v13 > v8 );
          *(_DWORD *)(a1 + 8) = v8;
        }
      }
      ++v7;
    }
    v18 = v8;
    v17 = 4LL * v8;
  }
  else
  {
    v17 = 4;
    v18 = 1;
  }
  *(_QWORD *)(a1 + 56) = v18;
  v19 = (void *)sub_2207820(v17);
  *(_QWORD *)(a1 + 48) = v19;
  memset(v19, 0, 4LL * *(_QWORD *)(a1 + 56));
  v20 = *(void **)(a1 + 72);
  *(_QWORD *)(a1 + 64) = 0;
  if ( !v20 )
  {
    *(_QWORD *)(a1 + 80) = v18;
    v22 = sub_2207820(v17);
    *(_QWORD *)(a1 + 72) = v22;
    v20 = (void *)v22;
  }
  memset(v20, 0, 4LL * *(_QWORD *)(a1 + 80));
  result = *(unsigned int *)(a1 + 8);
  *(_QWORD *)(a1 + 88) = 0;
  if ( (_DWORD)result )
  {
    result = **(unsigned int **)(a1 + 24);
    *(_DWORD *)(a1 + 40) = result;
  }
  return result;
}
