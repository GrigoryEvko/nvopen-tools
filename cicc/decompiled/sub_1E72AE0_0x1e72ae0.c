// Function: sub_1E72AE0
// Address: 0x1e72ae0
//
__int64 __fastcall sub_1E72AE0(__int64 a1, __int64 a2)
{
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 result; // rax
  __int64 (*v9)(); // rax
  __int64 v10; // rax

  *(_QWORD *)(a1 + 128) = a2;
  *(_QWORD *)(a1 + 16) = a2 + 632;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
  sub_1E726A0(a1 + 32, a2, a2 + 632);
  sub_1E72840(a1 + 136, *(_QWORD *)(a1 + 128), *(_QWORD *)(a1 + 16), a1 + 32, v3, v4);
  v5 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 512) = 0;
  v6 = v5;
  v7 = v5 + 72;
  result = sub_1F4B690(v6);
  if ( !(_BYTE)result )
    v7 = 0;
  if ( !*(_QWORD *)(a1 + 288) )
  {
    v9 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 128) + 32LL) + 16LL) + 40LL);
    if ( v9 == sub_1D00B00 )
      BUG();
    v10 = v9();
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v10 + 760LL))(
               v10,
               v7,
               *(_QWORD *)(a1 + 128));
    *(_QWORD *)(a1 + 288) = result;
  }
  return result;
}
