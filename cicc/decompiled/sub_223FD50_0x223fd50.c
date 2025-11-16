// Function: sub_223FD50
// Address: 0x223fd50
//
__int64 __fastcall sub_223FD50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  int v6; // eax
  __int64 v7; // rbx
  __int64 v8; // rdi
  char v9; // r9
  __int64 v10; // rbx
  __int64 result; // rax
  int v12; // r9d
  __int64 v13; // r8
  __int64 v14; // r8

  v5 = a1 + 88;
  v6 = *(_DWORD *)(a1 + 64);
  v7 = *(_QWORD *)(a1 + 80);
  v8 = *(_QWORD *)(a1 + 72);
  v9 = v6;
  v10 = a2 + v7;
  result = v6 & 0x10;
  v12 = v9 & 8;
  if ( v8 == v5 )
    v13 = 15;
  else
    v13 = *(_QWORD *)(a1 + 88);
  v14 = a2 + v13;
  if ( a2 != v8 )
  {
    v10 += a3;
    a3 = 0;
    v14 = v10;
  }
  if ( v12 )
  {
    *(_QWORD *)(a1 + 8) = a2;
    *(_QWORD *)(a1 + 16) = a2 + a3;
    *(_QWORD *)(a1 + 24) = v10;
    if ( (_DWORD)result )
      return sub_223FAE0((_QWORD *)a1, a2, v14, a4);
  }
  else if ( (_DWORD)result )
  {
    result = sub_223FAE0((_QWORD *)a1, a2, v14, a4);
    *(_QWORD *)(a1 + 8) = v10;
    *(_QWORD *)(a1 + 16) = v10;
    *(_QWORD *)(a1 + 24) = v10;
  }
  return result;
}
