// Function: sub_17E26D0
// Address: 0x17e26d0
//
void __fastcall sub_17E26D0(char *a1, char *a2)
{
  char *v2; // r13
  __int64 v4; // r12
  char *v5; // rbx
  bool v6; // cc
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rax
  char *v12; // rbx
  __int64 v13; // rdi

  if ( a1 != a2 )
  {
    v2 = a1 + 8;
    while ( a2 != v2 )
    {
      v4 = *(_QWORD *)v2;
      v5 = v2;
      v2 += 8;
      v6 = *(_QWORD *)(v4 + 16) <= *(_QWORD *)(*(_QWORD *)a1 + 16LL);
      *((_QWORD *)v2 - 1) = 0;
      if ( v6 )
      {
        v11 = *((_QWORD *)v2 - 2);
        v12 = v2 - 16;
        if ( *(_QWORD *)(v4 + 16) <= *(_QWORD *)(v11 + 16) )
        {
          *((_QWORD *)v2 - 1) = v4;
          continue;
        }
        while ( 1 )
        {
          v13 = *((_QWORD *)v12 + 1);
          *(_QWORD *)v12 = 0;
          *((_QWORD *)v12 + 1) = v11;
          if ( v13 )
            j_j___libc_free_0(v13, 32);
          v11 = *((_QWORD *)v12 - 1);
          if ( *(_QWORD *)(v4 + 16) <= *(_QWORD *)(v11 + 16) )
            break;
          v12 -= 8;
        }
        v10 = *(_QWORD *)v12;
        *(_QWORD *)v12 = v4;
        if ( !v10 )
          continue;
      }
      else
      {
        v7 = (v5 - a1) >> 3;
        if ( v5 - a1 > 0 )
        {
          do
          {
            v8 = *((_QWORD *)v5 - 1);
            v9 = *(_QWORD *)v5;
            v5 -= 8;
            *(_QWORD *)v5 = 0;
            *((_QWORD *)v5 + 1) = v8;
            if ( v9 )
              j_j___libc_free_0(v9, 32);
            --v7;
          }
          while ( v7 );
        }
        v10 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v4;
        if ( !v10 )
          continue;
      }
      j_j___libc_free_0(v10, 32);
    }
  }
}
