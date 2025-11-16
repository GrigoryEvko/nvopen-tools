// Function: sub_134CC10
// Address: 0x134cc10
//
__int64 __fastcall sub_134CC10(__int64 a1, __int64 a2, unsigned int a3)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r14
  unsigned int v6; // r15d

  v3 = *(_QWORD **)(a1 + 48);
  v4 = *(_QWORD **)(a1 + 56);
  if ( v3 == v4 )
  {
    return 7;
  }
  else
  {
    v6 = 7;
    while ( 1 )
    {
      v6 &= (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)*v3 + 40LL))(*v3, a2, a3);
      if ( (v6 & 3) == 0 )
        break;
      if ( v4 == ++v3 )
        return v6;
    }
    return 4;
  }
}
