// Function: sub_DF8ED0
// Address: 0xdf8ed0
//
__int64 __fastcall sub_DF8ED0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_WORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 50) = 0;
  v2 = (_QWORD *)sub_AA48A0(**(_QWORD **)(a2 + 32));
  v3 = sub_BCB2D0(v2);
  *(_QWORD *)(a1 + 32) = v3;
  result = sub_ACD640(v3, 1, 0);
  *(_QWORD *)(a1 + 40) = result;
  return result;
}
