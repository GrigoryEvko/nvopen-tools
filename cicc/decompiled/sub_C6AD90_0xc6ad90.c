// Function: sub_C6AD90
// Address: 0xc6ad90
//
_BYTE *__fastcall sub_C6AD90(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rdi
  _BYTE *result; // rax

  v2 = *(unsigned int *)(a1 + 8);
  *(_DWORD *)(a1 + 172) -= *(_DWORD *)(a1 + 168);
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8 * v2 - 4) )
    sub_C6A6A0(a1);
  v3 = *(_QWORD *)(a1 + 160);
  result = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v3 + 24) )
  {
    result = (_BYTE *)sub_CB5D20(v3, 125);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = result + 1;
    *result = 125;
  }
  --*(_DWORD *)(a1 + 8);
  return result;
}
