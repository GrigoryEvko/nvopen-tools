// Function: sub_9CAE40
// Address: 0x9cae40
//
__int64 __fastcall sub_9CAE40(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // rdx
  unsigned int v3; // eax

  v2 = a1[66];
  if ( a2 >= (unsigned __int64)((a1[67] - v2) >> 3) || *(_BYTE *)(*(_QWORD *)(v2 + 8LL * a2) + 8LL) != 14 )
    return 0;
  v3 = sub_9C2A90((__int64)a1, a2, 0);
  return sub_9CAD80(a1, v3);
}
