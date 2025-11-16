// Function: sub_22C6250
// Address: 0x22c6250
//
__int64 __fastcall sub_22C6250(unsigned __int8 *a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 result; // rax

  v3 = *((_QWORD *)a1 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  result = *(_DWORD *)(v3 + 8) >> 8;
  if ( !(_DWORD)result )
    return sub_22C6020(a1, a2, a3);
  return result;
}
