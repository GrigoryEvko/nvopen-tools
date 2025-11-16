// Function: sub_EFDF30
// Address: 0xefdf30
//
__int64 __fastcall sub_EFDF30(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_CB6200(a2, *(unsigned __int8 **)a1, *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 8) = 0;
  return result;
}
