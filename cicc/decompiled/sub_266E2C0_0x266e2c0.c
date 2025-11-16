// Function: sub_266E2C0
// Address: 0x266e2c0
//
__int64 __fastcall sub_266E2C0(__int64 *a1, __int64 a2, __int64 *a3, _BYTE *a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx

  v4 = *a1;
  if ( *(_BYTE *)(*a1 + 96) )
    return *(_QWORD *)(v4 + 304);
  v5 = *a3;
  if ( v5 )
  {
    *a4 = 1;
    sub_250ED80(a1[1], *a1, v5, 1);
    v4 = *a1;
    return *(_QWORD *)(v4 + 304);
  }
  return v5;
}
