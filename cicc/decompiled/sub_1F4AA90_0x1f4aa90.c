// Function: sub_1F4AA90
// Address: 0x1f4aa90
//
__int64 *__fastcall sub_1F4AA90(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax

  v6 = sub_22077B0(24);
  if ( v6 )
  {
    *(_DWORD *)v6 = a2;
    *(_QWORD *)(v6 + 8) = a3;
    *(_QWORD *)(v6 + 16) = a4;
  }
  *a1 = v6;
  a1[2] = (__int64)sub_1F4A060;
  a1[3] = (__int64)sub_1F4A1F0;
  return a1;
}
