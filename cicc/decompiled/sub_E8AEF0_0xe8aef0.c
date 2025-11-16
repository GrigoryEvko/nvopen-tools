// Function: sub_E8AEF0
// Address: 0xe8aef0
//
char *__fastcall sub_E8AEF0(__int64 a1, char *a2)
{
  char *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi
  _BYTE *v5; // rsi
  char *v6; // [rsp+8h] [rbp-8h] BYREF

  result = a2;
  v3 = *(_QWORD *)(a1 + 296);
  v6 = a2;
  v4 = *(_QWORD *)(v3 + 24);
  v5 = *(_BYTE **)(v4 + 64);
  if ( v5 == *(_BYTE **)(v4 + 72) )
    return sub_E8AD60(v4 + 56, v5, &v6);
  if ( v5 )
  {
    *(_QWORD *)v5 = result;
    v5 = *(_BYTE **)(v4 + 64);
  }
  *(_QWORD *)(v4 + 64) = v5 + 8;
  return result;
}
