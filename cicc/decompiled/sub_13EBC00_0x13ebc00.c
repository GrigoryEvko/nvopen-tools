// Function: sub_13EBC00
// Address: 0x13ebc00
//
void __fastcall sub_13EBC00(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx

  if ( a1[4] )
  {
    v1 = sub_13E7A30(a1 + 4, *a1, a1[1], a1[3]);
    v2 = *(_QWORD *)(v1 + 288);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v1 + 296);
      *(_QWORD *)(v1 + 296) = v2;
      *(_QWORD *)(v1 + 288) = v3;
    }
  }
}
