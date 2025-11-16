// Function: sub_1A01450
// Address: 0x1a01450
//
__int64 __fastcall sub_1A01450(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rbx
  __int64 result; // rax

  while ( 1 )
  {
    v3 = sub_19FEFC0(a1, 15, 16);
    v6 = v3;
    if ( !v3 )
      break;
    sub_1A01450(*(_QWORD *)(v3 - 24), a2);
    a1 = *(_QWORD *)(v6 - 48);
  }
  result = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v4, v5);
    result = *(unsigned int *)(a2 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = a1;
  ++*(_DWORD *)(a2 + 8);
  return result;
}
