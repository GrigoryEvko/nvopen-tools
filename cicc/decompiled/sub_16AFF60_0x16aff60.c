// Function: sub_16AFF60
// Address: 0x16aff60
//
__int64 __fastcall sub_16AFF60(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(*(_QWORD *)(a1 + 160) + 8LL);
  *(_BYTE *)(a1 + 192) = result;
  return result;
}
