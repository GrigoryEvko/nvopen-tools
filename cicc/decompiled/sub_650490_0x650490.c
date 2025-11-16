// Function: sub_650490
// Address: 0x650490
//
__int64 __fastcall sub_650490(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 result; // rax

  v2 = (_QWORD *)sub_8787C0();
  result = sub_85EB10(*(_QWORD *)(a1 + 184));
  v2[1] = *(_QWORD *)(a2 + 24);
  *v2 = *(_QWORD *)(result + 128);
  *(_QWORD *)(result + 128) = v2;
  return result;
}
