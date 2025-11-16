// Function: sub_21545C0
// Address: 0x21545c0
//
__int64 __fastcall sub_21545C0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  __int64 result; // rax
  _BYTE *v9; // rax
  _BYTE *v10; // rax

  if ( !a5 || (result = 1, !*a5) )
  {
    v9 = *(_BYTE **)(a6 + 24);
    if ( (unsigned __int64)v9 >= *(_QWORD *)(a6 + 16) )
    {
      sub_16E7DE0(a6, 91);
    }
    else
    {
      *(_QWORD *)(a6 + 24) = v9 + 1;
      *v9 = 91;
    }
    sub_21544A0(a1, a2, a3, a6, 0);
    v10 = *(_BYTE **)(a6 + 24);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(a6 + 16) )
    {
      sub_16E7DE0(a6, 93);
      return 0;
    }
    else
    {
      *(_QWORD *)(a6 + 24) = v10 + 1;
      *v10 = 93;
      return 0;
    }
  }
  return result;
}
