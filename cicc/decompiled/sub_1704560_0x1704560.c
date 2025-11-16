// Function: sub_1704560
// Address: 0x1704560
//
__int64 __fastcall sub_1704560(_BYTE *a1)
{
  int v1; // eax
  __int64 result; // rax
  int v3; // eax

  v1 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
  if ( (_BYTE)v1 == 16 )
    v1 = *(unsigned __int8 *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
  result = (unsigned int)(v1 - 1);
  if ( (unsigned __int8)result <= 5u || a1[16] == 76 )
  {
    v3 = sub_15F24E0((__int64)a1);
    a1[17] &= 1u;
    return sub_15F2440((__int64)a1, v3);
  }
  else
  {
    a1[17] &= 1u;
  }
  return result;
}
