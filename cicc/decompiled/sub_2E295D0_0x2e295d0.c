// Function: sub_2E295D0
// Address: 0x2e295d0
//
void __fastcall sub_2E295D0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  unsigned int i; // ebx
  __int64 v7; // r9
  int v8; // eax
  __int64 v9; // rdi
  unsigned int v10; // r8d
  __int16 *v11; // rax
  __int16 *v12; // rdx
  int v13; // eax
  int v14; // esi
  unsigned __int16 j; // cx
  int v16; // eax

  if ( a3 != 1 )
  {
    for ( i = 1; i != a3; ++i )
    {
      while ( 1 )
      {
        v9 = a1[13];
        if ( *(_QWORD *)(v9 + 8LL * i) || *(_QWORD *)(a1[16] + 8LL * i) )
        {
          v7 = *(_QWORD *)(a2 + 24);
          v8 = *(_DWORD *)(v7 + 4LL * (i >> 5));
          if ( !_bittest(&v8, i) )
            break;
        }
        if ( a3 == ++i )
          return;
      }
      v10 = i;
      v11 = (__int16 *)(*(_QWORD *)(a1[12] + 56LL) + 2LL * *(unsigned int *)(*(_QWORD *)(a1[12] + 8LL) + 24LL * i + 8));
      v12 = v11 + 1;
      v13 = *v11;
      v14 = i + v13;
      if ( (_WORD)v13 )
      {
        for ( j = i + v13; ; j = v14 )
        {
          if ( j < a3
            && (*(_QWORD *)(v9 + 8LL * j) || *(_QWORD *)(a1[16] + 8LL * j))
            && ((*(_DWORD *)(v7 + 4LL * (j >> 5)) >> j) & 1) == 0 )
          {
            v10 = j;
          }
          v16 = *v12++;
          if ( !(_WORD)v16 )
            break;
          v14 += v16;
        }
      }
      sub_2E285B0(a1, v10, 0);
    }
  }
}
