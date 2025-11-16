// Function: sub_349DFE0
// Address: 0x349dfe0
//
bool __fastcall sub_349DFE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 (*v4)(); // rax
  int v6; // r14d
  int v7; // eax
  int v8; // edx
  int v9; // ecx
  bool result; // al

  v3 = *(_QWORD *)(a2 + 32);
  v4 = *(__int64 (**)())(**(_QWORD **)(v3 + 16) + 144LL);
  if ( v4 == sub_2C8F680 )
    BUG();
  v6 = *(_DWORD *)(v4() + 104);
  v7 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a3 + 680LL))(a3, v3);
  v8 = *(_DWORD *)(a1 + 8);
  v9 = v7;
  result = v8 != 0 && v6 != v8;
  if ( result )
    return v9 != v8;
  return result;
}
