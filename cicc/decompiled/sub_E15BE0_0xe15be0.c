// Function: sub_E15BE0
// Address: 0xe15be0
//
__int64 __fastcall sub_E15BE0(_BYTE *a1, __int64 a2)
{
  __int64 result; // rax

  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)a1 + 32LL))(a1);
  result = a1[9] & 0xC0;
  if ( (a1[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 40LL))(a1, a2);
  return result;
}
