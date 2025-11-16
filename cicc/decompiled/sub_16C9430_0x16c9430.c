// Function: sub_16C9430
// Address: 0x16c9430
//
__int64 __fastcall sub_16C9430(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // r13

  result = 1;
  v4 = *((unsigned int *)a1 + 2);
  if ( (_DWORD)v4 )
  {
    v5 = sub_16EC640(v4, *a1, 0, 0);
    sub_22410F0(a2, v5 - 1, 0);
    sub_16EC640(*((unsigned int *)a1 + 2), *a1, *a2, v5);
    return 0;
  }
  return result;
}
