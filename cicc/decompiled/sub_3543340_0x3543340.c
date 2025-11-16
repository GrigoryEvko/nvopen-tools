// Function: sub_3543340
// Address: 0x3543340
//
void __fastcall sub_3543340(__int64 a1, __int64 *a2)
{
  unsigned int v2; // edx
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned __int64 v6; // rdi

  v2 = *(_DWORD *)(a1 + 3472);
  if ( v2 > 0x10 )
  {
    v3 = *a2;
    v4 = *a2 + 88LL * *((unsigned int *)a2 + 2);
    if ( v4 == *a2 )
    {
LABEL_11:
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v5 = *a2;
      while ( *(int *)(v5 + 52) <= 2 && v2 >= *(_DWORD *)(v5 + 60) )
      {
        v5 += 88;
        if ( v5 == v4 )
        {
          do
          {
            v4 -= 88;
            v6 = *(_QWORD *)(v4 + 32);
            if ( v6 != v4 + 48 )
              _libc_free(v6);
            sub_C7D6A0(*(_QWORD *)(v4 + 8), 8LL * *(unsigned int *)(v4 + 24), 8);
          }
          while ( v3 != v4 );
          goto LABEL_11;
        }
      }
    }
  }
}
