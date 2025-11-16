// Function: sub_CF4FA0
// Address: 0xcf4fa0
//
__int64 __fastcall sub_CF4FA0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  unsigned int v5; // ebx
  _QWORD *v6; // r15
  unsigned int v7; // r12d
  _QWORD *v9; // [rsp+8h] [rbp-38h]

  v9 = *(_QWORD **)(a1 + 16);
  if ( *(_QWORD **)(a1 + 8) == v9 )
  {
    return 3;
  }
  else
  {
    v5 = a4;
    v6 = *(_QWORD **)(a1 + 8);
    v7 = 3;
    do
    {
      LOBYTE(v7) = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(*(_QWORD *)*v6 + 24LL))(
                     *v6,
                     a2,
                     a3,
                     v5)
                 & v7;
      if ( !(_BYTE)v7 )
        break;
      ++v6;
    }
    while ( v9 != v6 );
  }
  return v7;
}
