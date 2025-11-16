// Function: sub_B1AD90
// Address: 0xb1ad90
//
__int64 __fastcall sub_B1AD90(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi

  v3 = *a1;
  result = *((unsigned int *)a1 + 2);
  v5 = *a1 + 8 * result;
  if ( *a1 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v5 - 8);
      v5 -= 8;
      if ( v6 )
      {
        v7 = *(_QWORD *)(v6 + 24);
        if ( v7 != v6 + 40 )
          _libc_free(v7, a2);
        a2 = 80;
        result = j_j___libc_free_0(v6, 80);
      }
    }
    while ( v5 != v3 );
  }
  *((_DWORD *)a1 + 2) = 0;
  return result;
}
