// Function: sub_8E52B0
// Address: 0x8e52b0
//
void __fastcall sub_8E52B0(__int128 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5[5]; // [rsp+10h] [rbp-28h] BYREF

  v5[0] = a3;
  v5[2] = *((_QWORD *)&a1 + 1);
  v5[3] = a2;
  sub_8E5140(
    qword_4F60598,
    v5,
    31 * (31 * (((unsigned __int64)a1 >> 3) + 527) + (*((_QWORD *)&a1 + 1) >> 3)) + (a2 >> 3),
    31 * (((unsigned __int64)a1 >> 3) + 527),
    a4,
    a5,
    a1,
    a2);
}
