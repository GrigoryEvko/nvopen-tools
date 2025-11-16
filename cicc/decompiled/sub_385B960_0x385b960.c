// Function: sub_385B960
// Address: 0x385b960
//
__int64 __fastcall sub_385B960(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // rax

  v4 = a2;
  v5 = sub_14806B0(a3, a2, a1, 0, 0);
  v6 = 0;
  if ( !*(_WORD *)(v5 + 24) )
  {
    v7 = *(_QWORD *)(v5 + 32);
    v8 = *(_DWORD *)(v7 + 32);
    v9 = *(_QWORD *)(v7 + 24);
    if ( v8 > 0x40 )
      v9 = *(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6));
    if ( (v9 & (1LL << ((unsigned __int8)v8 - 1))) == 0 )
      return a1;
    return v4;
  }
  return v6;
}
