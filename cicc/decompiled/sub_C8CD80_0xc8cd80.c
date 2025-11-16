// Function: sub_C8CD80
// Address: 0xc8cd80
//
__int64 __fastcall sub_C8CD80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9

  *(_QWORD *)a1 = 0;
  v7 = *(_BYTE *)(a3 + 28);
  *(_BYTE *)(a1 + 28) = v7;
  if ( v7 )
  {
    *(_QWORD *)(a1 + 8) = a2;
  }
  else
  {
    v9 = 8LL * *(unsigned int *)(a3 + 16);
    v10 = malloc(v9, a2, a3, a4, a5, a6);
    if ( !v10 && (v9 || (v10 = malloc(1, a2, v11, v12, v13, v14)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    *(_QWORD *)(a1 + 8) = v10;
  }
  return sub_C8CD20(a1, a3);
}
