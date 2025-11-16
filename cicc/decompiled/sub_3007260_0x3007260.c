// Function: sub_3007260
// Address: 0x3007260
//
__int64 __fastcall sub_3007260(__int64 a1)
{
  __int64 v1; // rdi
  int v2; // eax
  __int64 v4; // [rsp-8h] [rbp-8h]

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(unsigned __int8 *)(v1 + 8);
  if ( (_BYTE)v2 == 12 )
  {
    *((_BYTE *)&v4 - 8) = 0;
    *(&v4 - 2) = *(_DWORD *)(v1 + 8) >> 8;
    return *(&v4 - 2);
  }
  else
  {
    if ( (unsigned int)(v2 - 17) > 1 )
      BUG();
    return sub_BCAE30(v1);
  }
}
