// Function: sub_10609F0
// Address: 0x10609f0
//
__int64 *__fastcall sub_10609F0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx

  v1 = sub_22077B0(48);
  v2 = v1;
  if ( v1 )
  {
    sub_E3FC50(v1);
    *(_DWORD *)(v2 + 40) = 257;
    *(_QWORD *)v2 = &off_497BFF8;
  }
  *a1 = v2;
  return a1;
}
