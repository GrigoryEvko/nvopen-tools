// Function: sub_1E16C90
// Address: 0x1e16c90
//
__int64 __fastcall sub_1E16C90(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // r13
  _BYTE *v11; // rsi
  int v12; // eax
  __int64 v13; // rcx
  __int64 result; // rax
  unsigned int v15; // ebx
  __int64 v16; // rax

  v6 = 40LL * a2;
  v7 = v6 + *(_QWORD *)(a1 + 32);
  if ( !*(_BYTE *)v7 && (*(_WORD *)(v7 + 2) & 0xFF0) != 0 )
  {
    v16 = *(_QWORD *)(a1 + 32) + 40LL * (unsigned int)sub_1E16AB0(a1, a2, a3, a4, a5, a6);
    *(_WORD *)(v16 + 2) &= 0xF00Fu;
    *(_WORD *)(v7 + 2) &= 0xF00Fu;
  }
  v8 = ~a2;
  v9 = sub_1E15BB0(a1);
  v10 = v9;
  if ( v9 )
  {
    v11 = (_BYTE *)(v6 + *(_QWORD *)(a1 + 32));
    if ( *v11 )
    {
      v12 = *(_DWORD *)(a1 + 40);
      v13 = (unsigned int)(v12 + v8);
      if ( !(v12 + v8) )
        goto LABEL_7;
    }
    else
    {
      sub_1E69A50(v9, v11);
      v12 = *(_DWORD *)(a1 + 40);
      v13 = (unsigned int)(v12 + v8);
      if ( !(v12 + v8) )
        goto LABEL_7;
    }
    sub_1E69AC0(v10, v6 + *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 32) + v6 + 40, v13);
    v12 = *(_DWORD *)(a1 + 40);
    goto LABEL_7;
  }
  v12 = *(_DWORD *)(a1 + 40);
  v15 = v12 + v8;
  if ( v15 )
  {
    memmove((void *)(v6 + *(_QWORD *)(a1 + 32)), (const void *)(*(_QWORD *)(a1 + 32) + v6 + 40), 40LL * v15);
    v12 = *(_DWORD *)(a1 + 40);
  }
LABEL_7:
  result = (unsigned int)(v12 - 1);
  *(_DWORD *)(a1 + 40) = result;
  return result;
}
