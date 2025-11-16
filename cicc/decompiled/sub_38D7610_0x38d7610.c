// Function: sub_38D7610
// Address: 0x38d7610
//
double __fastcall sub_38D7610(unsigned int a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  int *v4; // rbx
  int *v5; // r13
  char v6; // al
  char v7; // r14
  int v8; // r12d
  double v9; // xmm0_8
  double v11; // [rsp+8h] [rbp-38h]
  double v12; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 96) + 10LL * a1;
  v3 = *(_QWORD *)(a2 + 72);
  v4 = (int *)(v3 + 16LL * *(unsigned __int16 *)(v2 + 2));
  v5 = (int *)(v3 + 16LL * *(unsigned __int16 *)(v2 + 4));
  if ( v5 == v4 )
    return 1.0;
  v6 = 0;
  v7 = 0;
  v11 = v12;
  do
  {
    if ( *v4 )
    {
      v8 = *v4;
      v9 = (double)(int)sub_39FAC40((unsigned int)v4[1]) / (double)v8;
      if ( v7 )
        v9 = fmin(v9, v11);
      v11 = v9;
      v6 = 1;
      v7 = 1;
    }
    v4 += 4;
  }
  while ( v5 != v4 );
  if ( v6 )
    v12 = v11;
  if ( !v7 )
    return 1.0;
  else
    return 1.0 / v12;
}
