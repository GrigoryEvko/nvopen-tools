// Function: sub_24A3210
// Address: 0x24a3210
//
void __fastcall sub_24A3210(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 *v5; // rbx
  bool v6; // cc
  __int64 v7; // r15
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rax
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rdi

  if ( a1 != a2 )
  {
    v2 = a1 + 1;
    while ( a2 != v2 )
    {
      v4 = *v2;
      v5 = v2++;
      v6 = *(_QWORD *)(v4 + 16) <= *(_QWORD *)(*a1 + 16);
      *(v2 - 1) = 0;
      if ( v6 )
      {
        v11 = *(v2 - 2);
        v12 = v2 - 2;
        if ( *(_QWORD *)(v4 + 16) <= *(_QWORD *)(v11 + 16) )
        {
          *(v2 - 1) = v4;
          continue;
        }
        while ( 1 )
        {
          v13 = v12[1];
          *v12 = 0;
          v12[1] = v11;
          if ( v13 )
            j_j___libc_free_0(v13);
          v11 = *(v12 - 1);
          if ( *(_QWORD *)(v4 + 16) <= *(_QWORD *)(v11 + 16) )
            break;
          --v12;
        }
        v10 = *v12;
        *v12 = v4;
        if ( !v10 )
          continue;
      }
      else
      {
        v7 = v5 - a1;
        if ( (char *)v5 - (char *)a1 > 0 )
        {
          do
          {
            v8 = *(v5 - 1);
            v9 = *v5--;
            *v5 = 0;
            v5[1] = v8;
            if ( v9 )
              j_j___libc_free_0(v9);
            --v7;
          }
          while ( v7 );
        }
        v10 = *a1;
        *a1 = v4;
        if ( !v10 )
          continue;
      }
      j_j___libc_free_0(v10);
    }
  }
}
