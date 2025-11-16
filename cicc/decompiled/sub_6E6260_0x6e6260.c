// Function: sub_6E6260
// Address: 0x6e6260
//
__int64 __fastcall sub_6E6260(_QWORD *a1)
{
  __int64 result; // rax

  sub_6E2E50(0, (__int64)a1);
  *a1 = sub_72C930(0);
  result = *(_QWORD *)dword_4F07508;
  *(_QWORD *)((char *)a1 + 68) = *(_QWORD *)dword_4F07508;
  return result;
}
