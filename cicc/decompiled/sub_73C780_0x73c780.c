// Function: sub_73C780
// Address: 0x73c780
//
__int64 __fastcall sub_73C780(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  sub_724C70(a3, 6);
  *(_QWORD *)(a3 + 184) = a1;
  *(_BYTE *)(a3 + 176) = 5;
  result = sub_73C750();
  *(_QWORD *)(a3 + 128) = result;
  return result;
}
