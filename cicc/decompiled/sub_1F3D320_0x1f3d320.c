// Function: sub_1F3D320
// Address: 0x1f3d320
//
__int64 __fastcall sub_1F3D320(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  _QWORD v5[2]; // [rsp+0h] [rbp-10h] BYREF

  v5[0] = a2;
  v5[1] = a3;
  if ( (_BYTE)a2 )
    v3 = word_42F2F80[(unsigned __int8)(a2 - 14)];
  else
    v3 = sub_1F58D30(v5);
  LOBYTE(v3) = v3 == 1;
  return (unsigned int)(4 * v3 + 1);
}
