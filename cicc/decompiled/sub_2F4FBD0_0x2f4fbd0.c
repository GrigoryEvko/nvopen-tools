// Function: sub_2F4FBD0
// Address: 0x2f4fbd0
//
__int64 __fastcall sub_2F4FBD0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  a1[8] = 0;
  a1[9] = 0;
  a1[10] = 0;
  a1[1] = sub_2EB2140(a3, (__int64 *)&unk_501EAD0, a2) + 8;
  a1[2] = sub_2EB2140(a3, &qword_501EAF0, a2) + 8;
  a1[11] = sub_2EB2140(a3, &qword_501EB00, a2) + 8;
  a1[3] = sub_2EB2140(a3, &qword_5025C20, a2) + 8;
  a1[4] = sub_2EB2140(a3, (__int64 *)&unk_501EC10, a2) + 8;
  a1[5] = sub_2EB2140(a3, qword_501FE48, a2) + 8;
  a1[7] = sub_2EB2140(a3, &qword_50209B0, a2) + 8;
  a1[6] = sub_2EB2140(a3, &qword_50208B0, a2) + 8;
  a1[8] = sub_2EB2140(a3, &qword_501D128, a2) + 8;
  a1[9] = sub_2EB2140(a3, &qword_5025C28, a2) + 8;
  a1[10] = sub_2EB2140(a3, &qword_501E910, a2) + 8;
  a1[12] = *(_QWORD *)(sub_2EB2140(a3, &qword_5023430, a2) + 8);
  a1[13] = *(_QWORD *)(sub_2EB2140(a3, &qword_5024420, a2) + 8);
  result = sub_2EB2140(a3, &qword_502A660, a2) + 8;
  *a1 = result;
  return result;
}
