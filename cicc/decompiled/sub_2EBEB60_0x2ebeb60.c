// Function: sub_2EBEB60
// Address: 0x2ebeb60
//
__int64 *__fastcall sub_2EBEB60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 *result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax

  v2 = *(unsigned int *)(a2 + 8);
  if ( (int)v2 < 0 )
  {
    v6 = *(_QWORD *)(a2 + 32);
    v8 = *(_QWORD *)(a1 + 56) + 16 * (v2 & 0x7FFFFFFF);
    v4 = *(_QWORD *)(a2 + 24);
    v7 = *(_QWORD *)(v8 + 8);
    result = (__int64 *)(v8 + 8);
    if ( a2 != v7 )
      goto LABEL_3;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 304);
    v4 = *(_QWORD *)(a2 + 24);
    result = (__int64 *)(v3 + 8 * v2);
    v6 = *(_QWORD *)(a2 + 32);
    v7 = *result;
    if ( a2 != *result )
    {
LABEL_3:
      *(_QWORD *)(v4 + 32) = v6;
      goto LABEL_4;
    }
  }
  *result = v6;
LABEL_4:
  if ( !v6 )
    v6 = v7;
  *(_QWORD *)(v6 + 24) = v4;
  *(_QWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 32) = 0;
  return result;
}
