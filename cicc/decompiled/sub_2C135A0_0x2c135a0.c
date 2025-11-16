// Function: sub_2C135A0
// Address: 0x2c135a0
//
__int64 __fastcall sub_2C135A0(_DWORD *a1)
{
  __int64 result; // rax

  *(_DWORD *)(*(_QWORD *)a1 + 104LL) = a1[2];
  *(_QWORD *)(*(_QWORD *)a1 + 96LL) = *((_QWORD *)a1 + 2);
  *(_BYTE *)(*(_QWORD *)a1 + 108LL) = *((_BYTE *)a1 + 24);
  *(_BYTE *)(*(_QWORD *)a1 + 109LL) = *((_BYTE *)a1 + 25);
  result = *(_QWORD *)a1;
  *(_BYTE *)(*(_QWORD *)a1 + 110LL) = *((_BYTE *)a1 + 26);
  return result;
}
