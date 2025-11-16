// Function: sub_154C1A0
// Address: 0x154c1a0
//
__int64 __fastcall sub_154C1A0(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi

  j___libc_free_0(*(_QWORD *)(a1 + 240));
  if ( *(_DWORD *)(a1 + 204) )
  {
    v2 = *(unsigned int *)(a1 + 200);
    v3 = *(_QWORD *)(a1 + 192);
    if ( (_DWORD)v2 )
    {
      v4 = 8 * v2;
      v5 = 0;
      do
      {
        v6 = *(_QWORD *)(v3 + v5);
        if ( v6 != -8 && v6 )
        {
          _libc_free(v6);
          v3 = *(_QWORD *)(a1 + 192);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 192);
  }
  _libc_free(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 160));
  j___libc_free_0(*(_QWORD *)(a1 + 120));
  j___libc_free_0(*(_QWORD *)(a1 + 80));
  return j___libc_free_0(*(_QWORD *)(a1 + 40));
}
