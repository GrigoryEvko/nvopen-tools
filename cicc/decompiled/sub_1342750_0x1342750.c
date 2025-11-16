// Function: sub_1342750
// Address: 0x1342750
//
void __fastcall sub_1342750(__int64 a1, int a2)
{
  __int64 v3; // r13
  _QWORD *v4; // r12
  __int64 v5; // rbx
  _QWORD *v6; // rdi

  v3 = a1 + 9632;
  v4 = (_QWORD *)(a1 + 32);
  v5 = a1 + 6432;
  do
  {
    v6 = v4;
    v5 += 16;
    v4 += 4;
    sub_133F510(v6);
    *(_QWORD *)(v5 - 16) = 0;
    *(_QWORD *)(v5 - 8) = 0;
  }
  while ( v5 != v3 );
  *(_DWORD *)(a1 + 9648) = a2;
  *(_QWORD *)(a1 + 9632) = 0;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
}
