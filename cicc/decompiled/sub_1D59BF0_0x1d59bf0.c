// Function: sub_1D59BF0
// Address: 0x1d59bf0
//
__int64 *__fastcall sub_1D59BF0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    sub_1D91830(v1);
    *(_QWORD *)v2 = &off_49F9D68;
    *(_WORD *)(v2 + 50) = 257;
  }
  *a1 = v2;
  return a1;
}
