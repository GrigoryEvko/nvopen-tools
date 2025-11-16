// Function: sub_16AFED0
// Address: 0x16afed0
//
__int64 __fastcall sub_16AFED0(_QWORD *a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = (_QWORD *)a1[20];
  v2 = v1[1];
  *(_BYTE *)(a1[23] + 8LL) = *(_BYTE *)(*v1 + 8LL);
  result = a1[24];
  *(_BYTE *)(result + 8) = *(_BYTE *)(v2 + 8);
  return result;
}
