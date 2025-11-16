// Function: sub_7E1360
// Address: 0x7e1360
//
__int64 __fastcall sub_7E1360(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  char *v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rax
  char *v6; // rdx

  result = *(_QWORD *)(a1 + 200);
  v3 = 0;
  if ( result )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v3 += *(_QWORD *)(result + 128);
        v4 = *(_QWORD *)(*(_QWORD *)(result + 40) + 32LL);
        if ( (*(_BYTE *)(v4 + 177) & 8) == 0 )
          break;
        result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)result + 96LL) + 88LL);
      }
      v5 = *(_QWORD *)(v4 + 168);
      if ( *(_BYTE *)(v5 + 113) != 2 )
        break;
      result = *(_QWORD *)(v5 + 120);
    }
    result = *(_QWORD *)(a1 + 176);
    if ( result )
    {
      result = *(_QWORD *)(result + 104);
      v6 = &v3[result];
      v3 -= result;
      if ( (*(_BYTE *)(a1 + 192) & 1) == 0 )
        v3 = v6;
    }
    *a2 = v3;
  }
  else
  {
    *a2 = -1;
  }
  return result;
}
