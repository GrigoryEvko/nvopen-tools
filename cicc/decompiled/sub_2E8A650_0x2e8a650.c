// Function: sub_2E8A650
// Address: 0x2e8a650
//
unsigned __int64 __fastcall sub_2E8A650(__int64 a1, unsigned int a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // r13
  _BYTE *v7; // rsi
  int v8; // eax
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 result; // rax
  unsigned int v12; // ebx
  __int64 v13; // rax

  v2 = 40LL * a2;
  v3 = v2 + *(_QWORD *)(a1 + 32);
  if ( !*(_BYTE *)v3 && (*(_WORD *)(v3 + 2) & 0xFF0) != 0 )
  {
    v13 = *(_QWORD *)(a1 + 32) + 40LL * (unsigned int)sub_2E89F40(a1, a2);
    *(_WORD *)(v13 + 2) &= 0xF00Fu;
    *(_WORD *)(v3 + 2) &= 0xF00Fu;
  }
  v4 = ~a2;
  v5 = sub_2E866D0(a1);
  v6 = v5;
  if ( v5 )
  {
    v7 = (_BYTE *)(v2 + *(_QWORD *)(a1 + 32));
    if ( *v7 )
    {
      v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
      v9 = (unsigned int)(v8 + v4);
      if ( !(v8 + v4) )
        goto LABEL_7;
    }
    else
    {
      sub_2EBEB60(v5, v7);
      v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
      v9 = (unsigned int)(v8 + v4);
      if ( !(v8 + v4) )
        goto LABEL_7;
    }
    sub_2EBEBD0(v6, v2 + *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 32) + v2 + 40, v9);
    v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
    goto LABEL_7;
  }
  v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  v12 = v8 + v4;
  if ( v12 )
  {
    memmove((void *)(v2 + *(_QWORD *)(a1 + 32)), (const void *)(*(_QWORD *)(a1 + 32) + v2 + 40), 40LL * v12);
    v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  }
LABEL_7:
  v10 = (unsigned int)(v8 + 0xFFFFFF);
  *(_WORD *)(a1 + 40) = v10;
  result = v10 >> 16;
  *(_BYTE *)(a1 + 42) = result;
  return result;
}
