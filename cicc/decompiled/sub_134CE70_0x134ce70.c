// Function: sub_134CE70
// Address: 0x134ce70
//
__int64 __fastcall sub_134CE70(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r14
  unsigned int v4; // r12d

  v2 = *(_QWORD **)(a1 + 48);
  v3 = *(_QWORD **)(a1 + 56);
  if ( v2 == v3 )
  {
    return 63;
  }
  else
  {
    v4 = 63;
    do
    {
      v4 &= (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v2 + 56LL))(*v2, a2);
      if ( v4 == 4 )
        break;
      ++v2;
    }
    while ( v3 != v2 );
  }
  return v4;
}
