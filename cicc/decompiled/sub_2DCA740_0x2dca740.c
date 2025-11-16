// Function: sub_2DCA740
// Address: 0x2dca740
//
void __fastcall sub_2DCA740(unsigned int *a1, unsigned int *a2, __int64 a3)
{
  char *v3; // r13
  unsigned int v5; // ebx
  unsigned int v6; // ebx
  unsigned int v7; // r14d
  unsigned int *i; // r15
  unsigned int v9; // eax
  unsigned int v10; // ebx

  if ( a1 != a2 )
  {
    v3 = (char *)(a1 + 1);
    if ( a2 != a1 + 1 )
    {
      do
      {
        while ( 1 )
        {
          v6 = sub_2DCA6E0(*(_QWORD *)(a3 + 8), *(unsigned int *)v3);
          if ( v6 > (unsigned int)sub_2DCA6E0(*(_QWORD *)(a3 + 8), *a1) )
            break;
          v7 = *(_DWORD *)v3;
          for ( i = (unsigned int *)(v3 - 4); ; i[2] = v9 )
          {
            v10 = sub_2DCA6E0(*(_QWORD *)(a3 + 8), v7);
            if ( v10 <= (unsigned int)sub_2DCA6E0(*(_QWORD *)(a3 + 8), *i) )
              break;
            v9 = *i--;
          }
          v3 += 4;
          i[1] = v7;
          if ( a2 == (unsigned int *)v3 )
            return;
        }
        v5 = *(_DWORD *)v3;
        if ( a1 != (unsigned int *)v3 )
          memmove(a1 + 1, a1, v3 - (char *)a1);
        v3 += 4;
        *a1 = v5;
      }
      while ( a2 != (unsigned int *)v3 );
    }
  }
}
