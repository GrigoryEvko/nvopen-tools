// Function: sub_2FDCC60
// Address: 0x2fdcc60
//
bool __fastcall sub_2FDCC60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  int v8; // esi
  int v9; // esi
  bool result; // al
  unsigned __int64 v11; // rdx

  v3 = 0;
  v4 = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD *)(a3 + 32);
  v7 = *(_QWORD *)(v6 + 32);
  if ( !*(_BYTE *)(v4 + 40) )
  {
    v8 = *(_DWORD *)(v4 + 48);
    if ( v8 < 0 )
      v3 = sub_2EBEE90(*(_QWORD *)(v6 + 32), v8);
  }
  if ( *(_BYTE *)(v4 + 80) )
    return 0;
  v9 = *(_DWORD *)(v4 + 88);
  if ( v9 >= 0 )
    return 0;
  v11 = sub_2EBEE90(v7, v9);
  result = v11 != 0 && v3 != 0;
  if ( !result )
    return 0;
  if ( a3 != *(_QWORD *)(v3 + 24) )
    return *(_QWORD *)(v11 + 24) == a3;
  return result;
}
