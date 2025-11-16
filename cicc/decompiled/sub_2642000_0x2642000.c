// Function: sub_2642000
// Address: 0x2642000
//
void __fastcall sub_2642000(_QWORD *a1)
{
  unsigned __int64 v1; // r8

  v1 = a1[2];
  if ( v1 )
    *(_QWORD *)(v1 + 8) = 0;
  a1[2] = 0;
  a1[3] = a1 + 1;
  a1[4] = a1 + 1;
  a1[5] = 0;
  sub_2641CE0(v1);
}
