// Function: sub_85F170
// Address: 0x85f170
//
__int64 *__fastcall sub_85F170(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v6; // rdi

  v4 = *(_QWORD *)(a1 + 40);
  if ( v4 )
  {
    if ( *(_BYTE *)(v4 + 28) == 3 )
    {
      v6 = *(_QWORD *)(v4 + 32);
      if ( v6 )
      {
        if ( v6 != a2 )
        {
          sub_85F170();
          a3 = -1;
        }
      }
    }
  }
  return sub_85F0B0(4u, a1, a3);
}
