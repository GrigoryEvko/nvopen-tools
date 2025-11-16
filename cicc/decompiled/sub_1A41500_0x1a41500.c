// Function: sub_1A41500
// Address: 0x1a41500
//
__int64 __fastcall sub_1A41500(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v8; // r12
  unsigned __int8 v9; // al
  unsigned __int64 *v10; // r8
  __int64 v11; // r14
  unsigned __int64 *v12; // rax
  unsigned __int64 *v13; // rsi
  unsigned __int64 *v14; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 *v18; // rax
  unsigned __int64 *v19; // r8
  __int64 v20; // r14
  unsigned __int64 *v21; // rsi
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  unsigned __int64 *v24; // rcx
  unsigned __int64 *v25; // rax
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // rsi
  __int64 v28; // r15
  unsigned __int64 *v29; // r14
  unsigned __int64 *v30; // rax
  __int64 v31; // rax
  int v32; // r9d
  _QWORD v33[2]; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int64 *v34[5]; // [rsp+18h] [rbp-28h] BYREF

  v8 = a4;
  v9 = *(_BYTE *)(a4 + 16);
  v33[0] = a4;
  if ( v9 == 17 )
  {
    v10 = a2 + 21;
    v11 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 80LL);
    if ( v11 )
      v11 -= 24;
    v12 = (unsigned __int64 *)a2[22];
    v13 = a2 + 21;
    if ( !v12 )
      goto LABEL_5;
    do
    {
      while ( 1 )
      {
        v16 = v12[2];
        v17 = v12[3];
        if ( v12[4] >= v8 )
          break;
        v12 = (unsigned __int64 *)v12[3];
        if ( !v17 )
          goto LABEL_11;
      }
      v13 = v12;
      v12 = (unsigned __int64 *)v12[2];
    }
    while ( v16 );
LABEL_11:
    if ( v10 == v13 || v13[4] > v8 )
    {
LABEL_5:
      v34[0] = v33;
      v14 = sub_1A41440(a2 + 20, v13, v34);
      v8 = v33[0];
      v13 = v14;
    }
    sub_1A3EEA0(a1, v11, *(_QWORD *)(v11 + 48), (unsigned __int64 *)v8, (__int64)(v13 + 5), a6);
  }
  else if ( v9 == 77 )
  {
    v26 = (unsigned __int64 *)a2[22];
    v27 = a2 + 21;
    v28 = *(_QWORD *)(a4 + 40);
    v29 = v27;
    if ( !v26 )
      goto LABEL_34;
    do
    {
      if ( v26[4] < a4 )
      {
        v26 = (unsigned __int64 *)v26[3];
      }
      else
      {
        v29 = v26;
        v26 = (unsigned __int64 *)v26[2];
      }
    }
    while ( v26 );
    if ( v27 == v29 || v29[4] > a4 )
    {
LABEL_34:
      v34[0] = v33;
      v30 = sub_1A41440(a2 + 20, v29, v34);
      v8 = v33[0];
      v29 = v30;
    }
    v31 = sub_157EE30(v28);
    sub_1A3EEA0(a1, v28, v31, (unsigned __int64 *)v8, (__int64)(v29 + 5), v32);
  }
  else if ( v9 <= 0x17u )
  {
    sub_1A3EEA0(a1, *(_QWORD *)(a3 + 40), a3 + 24, (unsigned __int64 *)a4, 0, a3 + 24);
  }
  else
  {
    v18 = (unsigned __int64 *)a2[22];
    v19 = a2 + 21;
    v20 = *(_QWORD *)(a4 + 40);
    v21 = a2 + 21;
    if ( !v18 )
      goto LABEL_23;
    do
    {
      while ( 1 )
      {
        v22 = v18[2];
        v23 = v18[3];
        if ( v18[4] >= v8 )
          break;
        v18 = (unsigned __int64 *)v18[3];
        if ( !v23 )
          goto LABEL_21;
      }
      v21 = v18;
      v18 = (unsigned __int64 *)v18[2];
    }
    while ( v22 );
LABEL_21:
    if ( v19 == v21 || (v24 = (unsigned __int64 *)v8, v21[4] > v8) )
    {
LABEL_23:
      v34[0] = v33;
      v25 = sub_1A41440(a2 + 20, v21, v34);
      v24 = (unsigned __int64 *)v33[0];
      v21 = v25;
    }
    sub_1A3EEA0(a1, v20, *(_QWORD *)(v8 + 32), v24, (__int64)(v21 + 5), a6);
  }
  return a1;
}
