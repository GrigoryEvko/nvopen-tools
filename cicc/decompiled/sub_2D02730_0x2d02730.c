// Function: sub_2D02730
// Address: 0x2d02730
//
__int64 __fastcall sub_2D02730(__int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi

  *(_QWORD *)a1 = off_4A25BD8;
  v2 = *(_QWORD *)(a1 + 176);
  if ( *(_DWORD *)(a1 + 188) )
  {
    v3 = *(unsigned int *)(a1 + 184);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v2 + v5);
        if ( v6 != (_QWORD *)-8LL && v6 )
        {
          sub_C7D6A0((__int64)v6, *v6 + 17LL, 8);
          v2 = *(_QWORD *)(a1 + 176);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  _libc_free(v2);
  return sub_BB9260(a1);
}
