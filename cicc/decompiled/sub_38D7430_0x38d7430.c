// Function: sub_38D7430
// Address: 0x38d7430
//
double __fastcall sub_38D7430(__int64 a1, _WORD *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned __int16 *v6; // rax
  unsigned __int16 *v7; // rdi
  double v8; // xmm2_8
  char v9; // dl
  char v10; // si
  int v11; // ecx
  double v12; // xmm0_8
  double v14; // [rsp+0h] [rbp-10h]

  v3 = (unsigned __int16)a2[1];
  v4 = *(_QWORD *)(a1 + 136);
  v5 = *(_QWORD *)(a1 + 160);
  v6 = (unsigned __int16 *)(v4 + 4 * v3);
  v7 = (unsigned __int16 *)(v4 + 4 * (v3 + (unsigned __int16)a2[2]));
  if ( v7 == v6 )
    return (double)(*a2 & 0x3FFF) / (double)*(int *)v5;
  v8 = v14;
  v9 = 0;
  v10 = 0;
  do
  {
    v11 = v6[1];
    if ( (_WORD)v11 )
    {
      v12 = (double)*(int *)(*(_QWORD *)(v5 + 32) + 32LL * *v6 + 8) / (double)v11;
      if ( v10 )
        v12 = fmin(v12, v8);
      v8 = v12;
      v9 = 1;
      v10 = 1;
    }
    v6 += 2;
  }
  while ( v7 != v6 );
  if ( v9 )
    v14 = v8;
  if ( !v10 )
    return (double)(*a2 & 0x3FFF) / (double)*(int *)v5;
  else
    return 1.0 / v14;
}
