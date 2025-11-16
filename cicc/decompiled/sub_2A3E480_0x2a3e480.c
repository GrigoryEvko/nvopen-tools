// Function: sub_2A3E480
// Address: 0x2a3e480
//
void __fastcall sub_2A3E480(char *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int *v8; // rcx
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  unsigned int *v15; // rax
  unsigned int *v16; // rdx
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  unsigned int v22; // eax
  double v23; // xmm1_8
  double v24; // xmm0_8
  double v25; // xmm0_8
  __int64 v26; // rax
  unsigned int v27[9]; // [rsp+Ch] [rbp-24h] BYREF

  v8 = a2;
  if ( a5 )
  {
    v10 = 0;
    v11 = 0;
    v12 = 0xFFFFFFFFLL;
    v13 = 0;
    do
    {
      v14 = *(unsigned int *)(a4 + 4 * v10);
      if ( v14 > v13 )
      {
        v11 = v10;
        v13 = *(unsigned int *)(a4 + 4 * v10);
      }
      if ( v12 > v14 )
        v12 = *(unsigned int *)(a4 + 4 * v10);
      ++v10;
    }
    while ( v10 != a5 );
    v15 = &a2[v11];
  }
  else
  {
    v15 = a2;
    v13 = 0;
    v12 = 0xFFFFFFFFLL;
  }
  v16 = &a2[a3];
  v17 = *v15;
  v18 = 0;
  if ( v16 != a2 )
  {
    do
    {
      v19 = *v8++;
      v18 += v19;
    }
    while ( v8 != v16 );
  }
  v27[0] = sub_F02DD0(v13, v13 + (a3 - 1) * v12);
  v20 = sub_F02E20(v27, v18);
  v21 = sub_BD5C60((__int64)a1);
  v22 = sub_B6E960(v21);
  if ( (unsigned int)qword_500AF08 >= v22 )
    v22 = qword_500AF08;
  if ( v22 > 0x63 )
  {
    v23 = 0.01000000000000001;
  }
  else
  {
    if ( !v22 )
      goto LABEL_15;
    v23 = 1.0 - (double)(int)v22 / 100.0;
  }
  if ( (v20 & 0x8000000000000000LL) != 0LL )
    v24 = (double)(int)(v20 & 1 | (v20 >> 1)) + (double)(int)(v20 & 1 | (v20 >> 1));
  else
    v24 = (double)(int)v20;
  v25 = v24 * v23;
  if ( v25 >= 9.223372036854776e18 )
  {
    v20 = (unsigned int)(int)(v25 - 9.223372036854776e18) ^ 0x8000000000000000LL;
LABEL_15:
    if ( v20 <= v17 )
      return;
LABEL_22:
    v26 = sub_BD5C60((__int64)a1);
    sub_2A3DFF0(a1, v26, v17, v18);
    return;
  }
  if ( (unsigned int)(int)v25 > v17 )
    goto LABEL_22;
}
