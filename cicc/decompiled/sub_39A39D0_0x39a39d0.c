// Function: sub_39A39D0
// Address: 0x39a39d0
//
__int64 __fastcall sub_39A39D0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v5; // eax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 200) + 4513LL) )
  {
    sub_39A35E0(a1, a2, 11, 251);
    v5 = sub_39BFF80(*(_QWORD *)(a1 + 200) + 5512LL, a3, 0);
    return sub_39A35E0(a1, a2, 7937, v5);
  }
  else
  {
    sub_39A35E0(a1, a2, 11, 3);
    return sub_39A39C0(a1, a2, 15, a3);
  }
}
