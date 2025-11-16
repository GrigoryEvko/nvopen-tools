// Function: sub_A170D0
// Address: 0xa170d0
//
__int64 __fastcall sub_A170D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  __int64 v5; // r12

  result = *(unsigned int *)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 12);
  v4 = result + 1;
  if ( a2 < 0 )
  {
    v5 = -2 * a2 + 1;
    if ( v3 >= v4 )
      goto LABEL_3;
LABEL_5:
    sub_C8D5F0(a1, a1 + 16, v4, 8);
    result = *(unsigned int *)(a1 + 8);
    goto LABEL_3;
  }
  v5 = 2 * a2;
  if ( v3 < v4 )
    goto LABEL_5;
LABEL_3:
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = v5;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
