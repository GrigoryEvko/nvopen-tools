// Function: sub_2FF63B0
// Address: 0x2ff63b0
//
__int64 *__fastcall sub_2FF63B0(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax

  v6 = sub_22077B0(0x18u);
  if ( v6 )
  {
    *(_DWORD *)v6 = a2;
    *(_QWORD *)(v6 + 8) = a3;
    *(_QWORD *)(v6 + 16) = a4;
  }
  *a1 = v6;
  a1[2] = (__int64)sub_2FF5590;
  a1[3] = (__int64)sub_2FF6000;
  return a1;
}
