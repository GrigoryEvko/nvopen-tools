// Function: sub_1D01030
// Address: 0x1d01030
//
__int64 __fastcall sub_1D01030(__int64 a1)
{
  _QWORD *v1; // rdx
  _QWORD *v2; // rcx
  _BYTE *v3; // rax
  unsigned int v4; // r8d

  v1 = *(_QWORD **)(a1 + 32);
  v2 = &v1[2 * *(unsigned int *)(a1 + 40)];
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
        v4 = v3[228] & 1;
        if ( (v3[228] & 1) != 0 && *(_WORD *)(*(_QWORD *)v3 + 24LL) == 47 )
          break;
      }
      v1 += 2;
      if ( v2 == v1 )
        return 0;
    }
  }
  return v4;
}
