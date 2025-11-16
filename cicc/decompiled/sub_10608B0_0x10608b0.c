// Function: sub_10608B0
// Address: 0x10608b0
//
__int64 *__fastcall sub_10608B0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx

  v1 = sub_22077B0(48);
  v2 = v1;
  if ( v1 )
  {
    sub_E3FC50(v1);
    *(_QWORD *)v2 = &off_49E5E00;
    *(_WORD *)(v2 + 42) = 257;
  }
  *a1 = v2;
  return a1;
}
