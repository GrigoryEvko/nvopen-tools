// Function: sub_CF51C0
// Address: 0xcf51c0
//
__int64 __fastcall sub_CF51C0(__int64 a1, __int64 a2, unsigned int a3)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r14
  unsigned int v6; // r15d

  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(_QWORD **)(a1 + 16);
  if ( v3 == v4 )
  {
    return 3;
  }
  else
  {
    v6 = 3;
    do
    {
      LOBYTE(v6) = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)*v3 + 32LL))(*v3, a2, a3) & v6;
      if ( !(_BYTE)v6 )
        break;
      ++v3;
    }
    while ( v4 != v3 );
  }
  return v6;
}
