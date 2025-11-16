// Function: sub_CF4B80
// Address: 0xcf4b80
//
__int64 __fastcall sub_CF4B80(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 result; // rax

  *a1 = *a2;
  v2 = a2[1];
  a2[1] = 0;
  a1[1] = v2;
  v3 = a2[2];
  a2[2] = 0;
  a1[2] = v3;
  v4 = a2[3];
  a2[3] = 0;
  a1[3] = v4;
  v5 = a2[4];
  a2[4] = 0;
  a1[4] = v5;
  v6 = a2[5];
  a2[5] = 0;
  a1[5] = v6;
  result = a2[6];
  a2[6] = 0;
  a1[6] = result;
  return result;
}
