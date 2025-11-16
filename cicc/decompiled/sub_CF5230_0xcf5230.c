// Function: sub_CF5230
// Address: 0xcf5230
//
__int64 __fastcall sub_CF5230(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r14
  unsigned int v6; // r15d

  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(_QWORD **)(a1 + 16);
  if ( v3 == v4 )
  {
    return 255;
  }
  else
  {
    v6 = 255;
    while ( 1 )
    {
      v6 &= (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v3 + 40LL))(*v3, a2, a3);
      if ( !v6 )
        break;
      if ( v4 == ++v3 )
        return v6;
    }
    return 0;
  }
}
