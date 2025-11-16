// Function: sub_1347320
// Address: 0x1347320
//
bool __fastcall sub_1347320(__int64 a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rcx
  unsigned __int64 v3; // rax

  v1 = *(unsigned int *)(a1 + 5640);
  v2 = *(_QWORD *)(a1 + 1368) - *(_QWORD *)(a1 + 5664);
  if ( (_DWORD)v1 != -1 )
  {
    v3 = *(_QWORD *)(a1 + 1360);
    if ( v3 <= 0xFFFFFFFFFFFFLL )
    {
      if ( v2 > (v1 * v3) >> 16 )
        return 1;
    }
    else if ( v2 > v1 * (v3 >> 16) )
    {
      return 1;
    }
  }
  return sub_1347290(a1);
}
