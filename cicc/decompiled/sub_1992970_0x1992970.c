// Function: sub_1992970
// Address: 0x1992970
//
__int64 __fastcall sub_1992970(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // r8
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx

  v1 = *(_QWORD *)(a1 + 368);
  v2 = 1;
  v3 = v1 + 1984LL * *(unsigned int *)(a1 + 376);
  if ( v1 != v3 )
  {
    while ( 1 )
    {
      v4 = *(unsigned int *)(v1 + 752);
      if ( v4 > 0xFFFE )
        break;
      v2 *= v4;
      if ( v2 <= 0x3FFFB )
      {
        v1 += 1984;
        if ( v3 != v1 )
          continue;
      }
      return v2;
    }
    return 262140;
  }
  return v2;
}
