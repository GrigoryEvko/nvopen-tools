// Function: sub_35C97A0
// Address: 0x35c97a0
//
__int64 __fastcall sub_35C97A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned int v7; // r11d
  unsigned int v8; // edi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  _DWORD *v13; // rax
  _DWORD *v14; // r9
  unsigned int v15; // ecx
  unsigned int v16; // edx
  int v17; // r8d
  unsigned int v18; // esi
  unsigned __int64 v19; // r13
  __int64 v20; // rbx
  void *v21; // rax
  void *v22; // rdi
  __int64 result; // rax
  __int64 v24; // rax

  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)a1 = &unk_4A3A778;
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  if ( a2 && (v5 = *(_QWORD *)(a2 + 104)) != 0 )
  {
    v7 = 0;
    v8 = 1;
    while ( 1 )
    {
      v9 = v5 + 10LL * v7;
      v10 = *(unsigned __int16 *)(v9 + 2);
      v11 = *(unsigned __int16 *)(v9 + 4);
      if ( ((unsigned __int16)v11 & (unsigned __int16)v10) == -1 )
        break;
      v12 = *(_QWORD *)(a2 + 80);
      v13 = (_DWORD *)(v12 + 24 * v10);
      v14 = (_DWORD *)(v12 + 24 * v11);
      if ( v14 != v13 )
      {
        v15 = 0;
        v16 = 0;
        do
        {
          v17 = v13[4];
          v18 = v16 + *v13;
          if ( v15 < v18 )
            v15 = v16 + *v13;
          v16 += v17;
          if ( v17 < 0 )
            v16 = v18;
          v13 += 6;
        }
        while ( v13 != v14 );
        if ( v15 > v8 )
        {
          do
            v8 *= 2;
          while ( v15 > v8 );
          *(_DWORD *)(a1 + 8) = v8;
        }
      }
      ++v7;
    }
    v20 = v8;
    v19 = 8LL * v8;
  }
  else
  {
    v19 = 8;
    v20 = 1;
  }
  *(_QWORD *)(a1 + 56) = v20;
  v21 = (void *)sub_2207820(v19);
  *(_QWORD *)(a1 + 48) = v21;
  memset(v21, 0, 8LL * *(_QWORD *)(a1 + 56));
  v22 = *(void **)(a1 + 72);
  *(_QWORD *)(a1 + 64) = 0;
  if ( !v22 )
  {
    *(_QWORD *)(a1 + 80) = v20;
    v24 = sub_2207820(v19);
    *(_QWORD *)(a1 + 72) = v24;
    v22 = (void *)v24;
  }
  memset(v22, 0, 8LL * *(_QWORD *)(a1 + 80));
  result = *(unsigned int *)(a1 + 8);
  *(_QWORD *)(a1 + 88) = 0;
  if ( (_DWORD)result )
  {
    result = **(unsigned int **)(a1 + 24);
    *(_DWORD *)(a1 + 40) = result;
  }
  return result;
}
