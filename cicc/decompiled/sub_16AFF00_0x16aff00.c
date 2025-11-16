// Function: sub_16AFF00
// Address: 0x16aff00
//
__int64 __fastcall sub_16AFF00(_QWORD *a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = (_QWORD *)a1[20];
  v2 = a1[24];
  *(_BYTE *)(*v1 + 8LL) = *(_BYTE *)(a1[23] + 8LL);
  result = v1[1];
  *(_BYTE *)(result + 8) = *(_BYTE *)(v2 + 8);
  return result;
}
