// Function: sub_72C970
// Address: 0x72c970
//
__int64 __fastcall sub_72C970(__int64 a1)
{
  __int64 result; // rax

  sub_724C70(a1, 0);
  result = sub_72C930();
  *(_QWORD *)(a1 + 128) = result;
  return result;
}
