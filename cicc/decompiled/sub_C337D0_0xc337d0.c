// Function: sub_C337D0
// Address: 0xc337d0
//
__int64 __fastcall sub_C337D0(__int64 a1)
{
  __int64 result; // rax

  result = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) + 64) >> 6;
  if ( !(_DWORD)result )
    return 1;
  return result;
}
