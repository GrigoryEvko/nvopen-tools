// Function: sub_14DE720
// Address: 0x14de720
//
__int64 __fastcall sub_14DE720(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // r15
  unsigned int v6; // r12d
  __int64 v7; // r13
  unsigned int v8; // eax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v11 = *a1;
  if ( dword_4F9D900 && a3 )
  {
    v5 = a4;
    v6 = 0;
    do
    {
      v7 = *(_QWORD *)(v11 + 16LL * v6 + 8);
      if ( !(unsigned __int8)sub_14DE6E0((__int64)a1, v7, a4, v5) )
        break;
      v5 -= v7;
      ++v6;
      v8 = a3;
      if ( dword_4F9D900 <= a3 )
        v8 = dword_4F9D900;
    }
    while ( v8 > v6 );
  }
  else
  {
    return 0;
  }
  return v6;
}
