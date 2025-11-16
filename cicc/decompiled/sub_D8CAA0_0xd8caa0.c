// Function: sub_D8CAA0
// Address: 0xd8caa0
//
_BYTE *__fastcall sub_D8CAA0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  _QWORD *v5; // r13
  unsigned __int8 *v6; // rax
  size_t v7; // rdx
  _BYTE *result; // rax

  v3 = sub_D8C9C0((__int64)a1, a2, a3);
  v4 = *a1;
  v5 = (_QWORD *)v3;
  v6 = (unsigned __int8 *)sub_BD5D20(*a1);
  sub_D88690(v5, a2, v6, v7, v4);
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
