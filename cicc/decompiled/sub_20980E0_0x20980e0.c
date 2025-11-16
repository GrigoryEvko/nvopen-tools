// Function: sub_20980E0
// Address: 0x20980e0
//
void __fastcall sub_20980E0(__int64 a1, __int64 a2, _QWORD *a3, int a4, unsigned int a5)
{
  unsigned int *v7; // rbx
  unsigned int *v8; // r12
  _BYTE *v9; // rax
  unsigned int v11; // [rsp-3Ch] [rbp-3Ch]

  if ( a4 )
  {
    sub_16E8750(a1, a5);
    sub_2097DA0(a2, a1, a3);
    v7 = *(unsigned int **)(a2 + 32);
    v8 = &v7[10 * *(unsigned int *)(a2 + 56)];
    if ( v7 != v8 )
    {
      v11 = a5 + 2;
      do
      {
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2]) != 1 )
        {
          v9 = *(_BYTE **)(a1 + 24);
          if ( (unsigned __int64)v9 >= *(_QWORD *)(a1 + 16) )
          {
            sub_16E7DE0(a1, 10);
          }
          else
          {
            *(_QWORD *)(a1 + 24) = v9 + 1;
            *v9 = 10;
          }
          sub_20980E0(a1, *(_QWORD *)v7, a3, (unsigned int)(a4 - 1), v11);
        }
        v7 += 10;
      }
      while ( v8 != v7 );
    }
  }
}
