// Function: sub_2166330
// Address: 0x2166330
//
__int64 __fastcall sub_2166330(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int v6; // r13d
  int i; // r14d
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // [rsp+Ch] [rbp-34h]

  if ( a2 > 3 )
  {
    result = 1;
    if ( a2 - 6 > 1 )
      return result;
  }
  else
  {
    result = 1;
    if ( a2 <= 1 )
      return result;
  }
  v12 = *(_QWORD *)(a3 + 32);
  if ( v12 <= 0 )
    return 0;
  v6 = 0;
  for ( i = 0; i != v12; ++i )
  {
    v8 = a3;
    if ( *(_BYTE *)(a3 + 8) == 16 )
      v8 = **(_QWORD **)(a3 + 16);
    v9 = sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v8, a4);
    v11 = a3;
    if ( *(_BYTE *)(a3 + 8) == 16 )
      v11 = **(_QWORD **)(a3 + 16);
    v6 += sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v11, v10) + v9;
  }
  return v6;
}
