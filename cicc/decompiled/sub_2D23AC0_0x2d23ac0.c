// Function: sub_2D23AC0
// Address: 0x2d23ac0
//
__int64 __fastcall sub_2D23AC0(__int64 a1, _QWORD *a2, int a3, unsigned int a4, int *a5)
{
  unsigned int v5; // r9d
  _QWORD *v7; // r14
  _QWORD *v8; // r15
  int v9; // ebx
  _QWORD *v10; // rsi
  __int64 v11; // rax
  __int64 v13; // r12
  unsigned int *v14; // rax
  __int64 v15; // r11
  __int64 v16; // rdx
  unsigned int *v17; // r8
  _QWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx

  v5 = 0;
  if ( (*(_QWORD *)(*a2 + 8LL * (a4 >> 6)) & (1LL << a4)) != 0 )
  {
    v7 = a2 + 9;
    v8 = a2 + 17;
    v9 = *a5;
    v10 = a2 + 9;
    if ( a3 == 1 )
      v10 = v8;
    v11 = *v10 + 24LL * a4;
    if ( v9 == *(_DWORD *)v11 )
    {
      v13 = *((_QWORD *)a5 + 1);
      if ( v13 == *(_QWORD *)(v11 + 8) )
      {
        v14 = (unsigned int *)sub_2D22AD0(a1, a4);
        v17 = &v14[v16];
        if ( v14 == v17 )
        {
          return 1;
        }
        else
        {
          v18 = v8;
          if ( a3 != 1 )
            v18 = v7;
          while ( 1 )
          {
            v19 = *v14;
            if ( (*(_QWORD *)(v15 + 8LL * ((unsigned int)v19 >> 6)) & (1LL << v19)) == 0 )
              break;
            v20 = *v18 + 24 * v19;
            if ( v9 != *(_DWORD *)v20 || v13 != *(_QWORD *)(v20 + 8) )
              break;
            if ( v17 == ++v14 )
              return 1;
          }
          return 0;
        }
      }
    }
  }
  return v5;
}
