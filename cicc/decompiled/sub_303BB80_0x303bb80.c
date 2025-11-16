// Function: sub_303BB80
// Address: 0x303bb80
//
__int64 __fastcall sub_303BB80(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5, int a6)
{
  __int64 v7; // r12
  unsigned int *v9; // rdx
  __int64 v11; // rsi
  __int128 v12; // rax
  int v13; // r9d
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  int v15; // [rsp+18h] [rbp-28h]

  v7 = a2;
  v9 = *(unsigned int **)(a2 + 40);
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v9 + 48LL) + 16LL * v9[2]) != 10 )
    return v7;
  v11 = *(_QWORD *)(a2 + 80);
  v14 = v11;
  if ( v11 )
  {
    sub_B96E90((__int64)&v14, v11, 1);
    v9 = *(unsigned int **)(v7 + 40);
  }
  v15 = *(_DWORD *)(v7 + 72);
  *(_QWORD *)&v12 = sub_33FAF80(a4, 233, (unsigned int)&v14, 12, 0, a6, *(_OWORD *)v9);
  v7 = sub_33FAF80(
         a4,
         *(_DWORD *)(v7 + 24),
         (unsigned int)&v14,
         *(unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * a3),
         *(_QWORD *)(*(_QWORD *)(v7 + 48) + 16LL * a3 + 8),
         v13,
         v12);
  if ( !v14 )
    return v7;
  sub_B91220((__int64)&v14, v14);
  return v7;
}
