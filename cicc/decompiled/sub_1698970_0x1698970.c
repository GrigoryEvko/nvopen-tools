// Function: sub_1698970
// Address: 0x1698970
//
__int64 __fastcall sub_1698970(_WORD *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // eax
  __int64 v4; // rdx
  unsigned int v5; // r15d
  _QWORD *v6; // r12
  _QWORD *v7; // r14
  __int64 v8; // rax
  int v9; // ebx
  int v10; // eax
  unsigned int v11; // ebx
  __int64 v12; // rdx
  int v13; // eax
  _QWORD *v14; // rsi
  unsigned int v15; // r12d
  int v16; // eax
  __int64 v18; // rax
  bool v19; // [rsp+Fh] [rbp-61h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  int v21; // [rsp+10h] [rbp-60h]
  _QWORD src[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = sub_1698470((__int64)a1);
  v20 = sub_16984A0(a2);
  v3 = sub_1698310((__int64)a1);
  v4 = v20;
  v5 = v3;
  if ( v3 > 2 )
  {
    v18 = sub_2207820(16LL * v3);
    v4 = v20;
    v7 = (_QWORD *)v18;
    v6 = (_QWORD *)(v18 + 8LL * v5);
    v19 = v18 != 0;
  }
  else
  {
    v19 = 1;
    v6 = &src[v3];
    v7 = src;
    if ( !v3 )
      goto LABEL_5;
  }
  v8 = 0;
  do
  {
    v7[v8] = *(_QWORD *)(v2 + 8 * v8);
    v6[v8] = *(_QWORD *)(v4 + 8 * v8);
    *(_QWORD *)(v2 + 8 * v8++) = 0;
  }
  while ( v5 > (unsigned int)v8 );
LABEL_5:
  a1[8] -= *(_WORD *)(a2 + 16);
  v9 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  v21 = v9;
  v10 = sub_16A7150(v6, v5, v4);
  v11 = v9 - 1;
  v12 = v11 - v10;
  if ( v11 != v10 )
  {
    a1[8] += v11 - v10;
    sub_16A7D00(v6);
  }
  v13 = sub_16A7150(v7, v5, v12);
  if ( v11 != v13 )
  {
    a1[8] -= v11 - v13;
    sub_16A7D00(v7);
  }
  if ( (int)sub_16A98D0(v7, v6, v5) < 0 )
  {
    --a1[8];
    sub_16A7D00(v7);
  }
  if ( v21 )
  {
    while ( 1 )
    {
      if ( (int)sub_16A98D0(v7, v6, v5) >= 0 )
      {
        sub_16A7520(v7, v6, 0, v5);
        sub_16A70D0(v2, v11);
      }
      sub_16A7D00(v7);
      if ( !v11 )
        break;
      --v11;
    }
  }
  v14 = v6;
  v15 = 3;
  v16 = sub_16A98D0(v7, v14, v5);
  if ( v16 <= 0 )
  {
    v15 = 2;
    if ( v16 )
      v15 = (unsigned __int8)sub_16A7080(v7, v5) ^ 1;
  }
  if ( v5 > 2 && v19 )
    j_j___libc_free_0_0(v7);
  return v15;
}
