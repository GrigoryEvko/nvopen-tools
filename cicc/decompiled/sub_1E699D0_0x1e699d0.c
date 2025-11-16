// Function: sub_1E699D0
// Address: 0x1e699d0
//
__int64 *__fastcall sub_1E699D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax

  v2 = *(unsigned int *)(a2 + 8);
  if ( (int)v2 >= 0 )
  {
    result = (__int64 *)(*(_QWORD *)(a1 + 272) + 8 * v2);
    v4 = *result;
    if ( *result )
      goto LABEL_3;
LABEL_7:
    *(_QWORD *)(a2 + 24) = a2;
    *(_QWORD *)(a2 + 32) = 0;
    *result = a2;
    return result;
  }
  v6 = *(_QWORD *)(a1 + 24) + 16 * (v2 & 0x7FFFFFFF);
  v4 = *(_QWORD *)(v6 + 8);
  result = (__int64 *)(v6 + 8);
  if ( !v4 )
    goto LABEL_7;
LABEL_3:
  v5 = *(_QWORD *)(v4 + 24);
  *(_QWORD *)(v4 + 24) = a2;
  *(_QWORD *)(a2 + 24) = v5;
  if ( (*(_BYTE *)(a2 + 3) & 0x10) != 0 )
  {
    *(_QWORD *)(a2 + 32) = v4;
    *result = a2;
  }
  else
  {
    *(_QWORD *)(a2 + 32) = 0;
    *(_QWORD *)(v5 + 32) = a2;
  }
  return result;
}
