// Function: sub_2E6C8E0
// Address: 0x2e6c8e0
//
_BYTE *__fastcall sub_2E6C8E0(__int64 a1)
{
  void *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  _BYTE *result; // rax

  v2 = sub_CB72A0();
  v3 = (__int64)v2;
  if ( a1 && (v4 = *(_QWORD *)a1) != 0 )
    sub_2E39560(v4, (__int64)v2);
  else
    sub_904010((__int64)v2, "nullptr");
  v5 = sub_904010(v3, " {");
  v6 = sub_CB59D0(v5, *(unsigned int *)(a1 + 72));
  v7 = sub_904010(v6, ", ");
  v8 = sub_CB59D0(v7, *(unsigned int *)(a1 + 76));
  result = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 24) )
    return (_BYTE *)sub_CB5D20(v8, 125);
  *(_QWORD *)(v8 + 32) = result + 1;
  *result = 125;
  return result;
}
