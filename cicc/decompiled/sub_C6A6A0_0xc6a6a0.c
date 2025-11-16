// Function: sub_C6A6A0
// Address: 0xc6a6a0
//
__int64 __fastcall sub_C6A6A0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 168);
  if ( (_DWORD)result )
  {
    sub_CB5D20(*(_QWORD *)(a1 + 160), 10);
    return sub_CB69B0(*(_QWORD *)(a1 + 160), *(unsigned int *)(a1 + 172));
  }
  return result;
}
