// Function: sub_26E0DD0
// Address: 0x26e0dd0
//
__int64 __fastcall sub_26E0DD0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r9
  size_t v6; // r10
  int *v7; // r11
  size_t v8; // r15
  unsigned __int64 v9; // r13
  _QWORD *v10; // rax
  size_t v11; // r8
  _QWORD *v12; // r12
  unsigned __int64 v13; // rsi
  int *v14; // rsi
  int v15; // eax
  __int64 v16; // r12
  __int64 result; // rax
  __int64 i; // r15
  size_t v19; // [rsp+8h] [rbp-108h]
  __int64 v20; // [rsp+10h] [rbp-100h]
  __int64 v21; // [rsp+18h] [rbp-F8h]
  int *v22; // [rsp+20h] [rbp-F0h]
  size_t v23; // [rsp+20h] [rbp-F0h]
  size_t v24; // [rsp+28h] [rbp-E8h]
  int *v25; // [rsp+28h] [rbp-E8h]
  __int64 v26; // [rsp+28h] [rbp-E8h]
  size_t v27[2]; // [rsp+30h] [rbp-E0h] BYREF
  int v28[52]; // [rsp+40h] [rbp-D0h] BYREF

  v4 = a2;
  v6 = *(_QWORD *)(a2 + 24);
  v7 = *(int **)(a2 + 16);
  v8 = v6;
  if ( v7 )
  {
    v22 = *(int **)(a2 + 16);
    v24 = *(_QWORD *)(a2 + 24);
    sub_C7D030(v28);
    sub_C7D280(v28, v22, v24);
    sub_C7D290(v28, v27);
    v8 = v27[0];
    v4 = a2;
    v7 = v22;
    v6 = v24;
  }
  v9 = a3[1];
  v10 = *(_QWORD **)(*a3 + 8 * (v8 % v9));
  v11 = v8 % v9;
  if ( !v10 || (v12 = (_QWORD *)*v10) == 0 )
  {
LABEL_17:
    v16 = *(_QWORD *)(v4 + 144);
    result = v4 + 128;
    v26 = v4 + 128;
    if ( v16 != v4 + 128 )
    {
      do
      {
        for ( i = *(_QWORD *)(v16 + 64); v16 + 48 != i; i = sub_220EF30(i) )
          sub_26E0DD0(a1, i + 48, a3, a4, v11);
        result = sub_220EF30(v16);
        v16 = result;
      }
      while ( v26 != result );
    }
    return result;
  }
  v13 = v12[3];
  a4 = 0;
  while ( v8 == v13 )
  {
    if ( v6 != v12[2] )
      break;
    v14 = (int *)v12[1];
    if ( v7 != v14 )
    {
      if ( !v7 )
        break;
      if ( !v14 )
        break;
      v19 = v11;
      v20 = v4;
      v21 = a4;
      v23 = v6;
      v25 = v7;
      v15 = memcmp(v7, v14, v6);
      v7 = v25;
      v6 = v23;
      a4 = v21;
      v4 = v20;
      v11 = v19;
      if ( v15 )
        break;
    }
    v12 = (_QWORD *)*v12;
    ++a4;
    if ( !v12 )
      goto LABEL_16;
LABEL_8:
    v13 = v12[3];
    if ( v11 != v13 % v9 )
      goto LABEL_16;
  }
  if ( a4 )
    goto LABEL_22;
  v12 = (_QWORD *)*v12;
  if ( v12 )
    goto LABEL_8;
LABEL_16:
  if ( !a4 )
    goto LABEL_17;
LABEL_22:
  result = *(_QWORD *)(v4 + 56);
  *(_QWORD *)(a1 + 424) += result;
  return result;
}
