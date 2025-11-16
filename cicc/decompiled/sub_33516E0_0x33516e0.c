// Function: sub_33516E0
// Address: 0x33516e0
//
__int64 __fastcall sub_33516E0(__int64 a1)
{
  _QWORD *v1; // rdx
  _QWORD *v2; // rcx
  _BYTE *v3; // rax
  unsigned int v4; // r8d

  v1 = *(_QWORD **)(a1 + 40);
  v2 = &v1[2 * *(unsigned int *)(a1 + 48)];
  if ( v1 == v2 )
  {
    return 0;
  }
  else
  {
    while ( 1 )
    {
      if ( (*v1 & 6) == 0 )
      {
        v3 = (_BYTE *)(*v1 & 0xFFFFFFFFFFFFFFF8LL);
        v4 = v3[248] & 1;
        if ( (v3[248] & 1) != 0 && *(_DWORD *)(*(_QWORD *)v3 + 24LL) == 50 )
          break;
      }
      v1 += 2;
      if ( v2 == v1 )
        return 0;
    }
  }
  return v4;
}
