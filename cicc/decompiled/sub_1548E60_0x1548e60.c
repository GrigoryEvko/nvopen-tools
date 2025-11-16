// Function: sub_1548E60
// Address: 0x1548e60
//
char __fastcall sub_1548E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  int v9; // r13d
  __int64 v10; // rdx
  __int64 v11; // r14
  char result; // al

  v7 = a1;
  v8 = *(_QWORD *)a1;
  v9 = *(_DWORD *)(a1 + 8);
  while ( 1 )
  {
    v10 = *(_QWORD *)(v7 - 16);
    v11 = v7;
    v7 -= 16;
    result = sub_1548C70(&a7, v8, v10);
    if ( !result )
      break;
    *(_QWORD *)(v7 + 16) = *(_QWORD *)v7;
    *(_DWORD *)(v7 + 24) = *(_DWORD *)(v7 + 8);
  }
  *(_QWORD *)v11 = v8;
  *(_DWORD *)(v11 + 8) = v9;
  return result;
}
