// Function: sub_222B6D0
// Address: 0x222b6d0
//
__int64 __fastcall sub_222B6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rdi
  char v8; // bl
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  char *v13; // rsi
  __int64 v14; // r8
  signed __int64 v15; // rbx
  __int64 v16; // r12
  signed __int64 v17; // rax
  signed __int64 v18; // rdx
  bool v19; // cc
  __int64 result; // rax
  __int64 v21; // rdx
  int v22; // ecx
  unsigned __int64 v23; // rcx

  v6 = *(_DWORD *)(a1 + 120);
  v7 = *(_QWORD *)(a1 + 200);
  if ( (v6 & 0x10) != 0 )
    LOBYTE(v6) = 1;
  v8 = v6 & 1;
  if ( !v7 )
    sub_426219(0, a2, 1, a4);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v7 + 48LL))(v7) || !v8 || *(_BYTE *)(a1 + 169) )
    return sub_22405B0(a1, a2, a3);
  v10 = *(_QWORD *)(a1 + 40);
  v11 = *(_QWORD *)(a1 + 48) - v10;
  if ( !*(_BYTE *)(a1 + 170) )
  {
    v12 = *(_QWORD *)(a1 + 160);
    if ( v12 > 1 )
      v11 = v12 - 1;
  }
  if ( v11 >= 1024 )
    v11 = 1024;
  if ( a3 < v11 )
    return sub_22405B0(a1, a2, a3);
  v13 = *(char **)(a1 + 32);
  v14 = a3;
  v15 = v10 - (_QWORD)v13;
  v16 = v10 - (_QWORD)v13 + a3;
  v17 = sub_2207E50((FILE **)(a1 + 104), v13, v10 - (_QWORD)v13, a2, v14);
  if ( v16 == v17 )
  {
    v21 = *(_QWORD *)(a1 + 152);
    v22 = *(_DWORD *)(a1 + 120);
    *(_QWORD *)(a1 + 8) = v21;
    *(_QWORD *)(a1 + 16) = v21;
    *(_QWORD *)(a1 + 24) = v21;
    if ( ((v22 & 0x10) != 0 || (v22 & 1) != 0) && (v23 = *(_QWORD *)(a1 + 160), v23 > 1) )
    {
      *(_QWORD *)(a1 + 40) = v21;
      *(_QWORD *)(a1 + 32) = v21;
      *(_QWORD *)(a1 + 48) = v21 + v23 - 1;
    }
    else
    {
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 48) = 0;
    }
    *(_BYTE *)(a1 + 170) = 1;
  }
  v18 = v17 - v15;
  v19 = v15 < v17;
  result = 0;
  if ( v19 )
    return v18;
  return result;
}
