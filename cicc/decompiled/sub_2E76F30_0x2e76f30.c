// Function: sub_2E76F30
// Address: 0x2e76f30
//
__int64 __fastcall sub_2E76F30(__int64 a1, __int64 *a2)
{
  __int64 result; // rax

  result = sub_B2D610(*a2, 20);
  if ( !(_BYTE)result )
    return (*(unsigned int (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 392LL))(a1, a2) ^ 1;
  return result;
}
