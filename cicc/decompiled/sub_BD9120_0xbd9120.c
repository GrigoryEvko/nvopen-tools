// Function: sub_BD9120
// Address: 0xbd9120
//
__int64 __fastcall sub_BD9120(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rdx
  __int64 result; // rax

  v2 = *a1;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 16LL);
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
    a2 = **(_QWORD **)(a2 + 16);
  if ( v3 == sub_BD8D60 )
    return 1;
  result = ((__int64 (__fastcall *)(__int64, __int64))v3)(v2, a2);
  if ( !BYTE1(result) )
    return 1;
  return result;
}
