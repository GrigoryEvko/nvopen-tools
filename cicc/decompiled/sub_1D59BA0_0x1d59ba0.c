// Function: sub_1D59BA0
// Address: 0x1d59ba0
//
__int64 *__fastcall sub_1D59BA0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    sub_1D91830(v1);
    *(_DWORD *)(v2 + 44) = 2;
    *(_QWORD *)v2 = &off_49F9D40;
    *(_BYTE *)(v2 + 52) = 1;
  }
  *a1 = v2;
  return a1;
}
