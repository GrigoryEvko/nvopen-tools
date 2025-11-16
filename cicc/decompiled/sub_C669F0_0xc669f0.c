// Function: sub_C669F0
// Address: 0xc669f0
//
__int64 __fastcall sub_C669F0(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  __int64 result; // rax

  sub_C66990(a1, a2, a3);
  result = sub_CB6200(*(_QWORD *)(a1 + 48), a2, a3);
  *(_QWORD *)(a1 + 64) = 0;
  return result;
}
