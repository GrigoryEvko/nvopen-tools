// Function: sub_1E729A0
// Address: 0x1e729a0
//
__int64 __fastcall sub_1E729A0(__int64 *a1, __int64 a2)
{
  int v3; // r8d
  int v4; // r9d
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 result; // rax
  __int64 (*v10)(); // rax
  __int64 v11; // rax
  __int64 (*v12)(); // rax
  __int64 v13; // rax

  a1[16] = a2;
  a1[2] = a2 + 632;
  a1[3] = *(_QWORD *)(a2 + 24);
  sub_1E726A0((__int64)(a1 + 4), a2, a2 + 632);
  sub_1E72840((__int64)(a1 + 18), a1[16], a1[2], (__int64)(a1 + 4), v3, v4);
  sub_1E72840((__int64)(a1 + 64), a1[16], a1[2], (__int64)(a1 + 4), v5, v6);
  v7 = a1[2];
  v8 = v7 + 72;
  result = sub_1F4B690(v7);
  if ( !(_BYTE)result )
    v8 = 0;
  if ( !a1[37] )
  {
    v10 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1[16] + 32) + 16LL) + 40LL);
    if ( v10 == sub_1D00B00 )
      BUG();
    v11 = v10();
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v11 + 760LL))(v11, v8, a1[16]);
    a1[37] = result;
  }
  if ( !a1[83] )
  {
    v12 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1[16] + 32) + 16LL) + 40LL);
    if ( v12 == sub_1D00B00 )
      BUG();
    v13 = v12();
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v13 + 760LL))(v13, v8, a1[16]);
    a1[83] = result;
  }
  a1[112] = 0;
  a1[118] = 0;
  return result;
}
