// Function: sub_854B90
// Address: 0x854b90
//
_QWORD *sub_854B90()
{
  __int64 v0; // rax
  _QWORD *v1; // r13
  _QWORD *v2; // rbx
  __int64 v3; // rdi

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v1 = *(_QWORD **)(v0 + 440);
  *(_QWORD *)(v0 + 440) = 0;
  if ( v1 )
  {
    v2 = v1;
    do
    {
      v3 = v2[8];
      if ( v3 )
      {
        sub_869FD0(v3, (unsigned int)dword_4F04C64);
        v2[8] = 0;
      }
      v2 = (_QWORD *)*v2;
    }
    while ( v2 );
  }
  return v1;
}
