// Function: sub_1CB8E10
// Address: 0x1cb8e10
//
void *__fastcall sub_1CB8E10(__int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi

  *(_QWORD *)a1 = off_49F8750;
  v2 = *(_QWORD *)(a1 + 160);
  if ( *(_DWORD *)(a1 + 172) )
  {
    v3 = *(unsigned int *)(a1 + 168);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD *)(v2 + v5);
        if ( v6 != -8 && v6 )
        {
          _libc_free(v6);
          v2 = *(_QWORD *)(a1 + 160);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  _libc_free(v2);
  return sub_1636790((_QWORD *)a1);
}
