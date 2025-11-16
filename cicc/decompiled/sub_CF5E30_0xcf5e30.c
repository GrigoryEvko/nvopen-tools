// Function: sub_CF5E30
// Address: 0xcf5e30
//
__int64 __fastcall sub_CF5E30(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r14
  unsigned int v4; // r12d

  v2 = *(_QWORD **)(a1 + 8);
  v3 = *(_QWORD **)(a1 + 16);
  if ( v2 == v3 )
  {
    return 255;
  }
  else
  {
    v4 = 255;
    while ( 1 )
    {
      v4 &= (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v2 + 48LL))(*v2, a2);
      if ( !v4 )
        break;
      if ( v3 == ++v2 )
        return v4;
    }
    return 0;
  }
}
