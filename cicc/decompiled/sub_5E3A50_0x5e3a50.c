// Function: sub_5E3A50
// Address: 0x5e3a50
//
__int64 __fastcall sub_5E3A50(__int64 a1, unsigned int a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rdi

  do
  {
    if ( *(_BYTE *)(a1 + 140) == 7 )
    {
      v9 = *(_QWORD *)(a1 + 168);
      v10 = *(_QWORD *)(v9 + 48);
      if ( v10 )
        sub_5E39A0(v10, *(_QWORD *)(v9 + 8), a2, a3, a5, a6);
      *(_BYTE *)(a1 + 142) |= 4u;
    }
    result = sub_8D48B0(a1, 0);
    a1 = result;
  }
  while ( result );
  return result;
}
