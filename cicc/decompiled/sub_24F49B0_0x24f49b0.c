// Function: sub_24F49B0
// Address: 0x24f49b0
//
__int64 __fastcall sub_24F49B0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v6; // rax
  _QWORD *v7; // r9
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rax
  _QWORD *v10; // r8
  _QWORD *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 *v14; // r13
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 result; // rax
  __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v23[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v24; // [rsp+20h] [rbp-30h]
  __int64 v25; // [rsp+28h] [rbp-28h]

  v6 = sub_B43CB0(a2);
  v7 = (_QWORD *)(a1 + 16);
  v8 = v6;
  v9 = *(_QWORD **)(a1 + 24);
  if ( v9 )
  {
    v10 = (_QWORD *)(a1 + 16);
    v11 = *(_QWORD **)(a1 + 24);
    do
    {
      while ( 1 )
      {
        v12 = v11[2];
        v13 = v11[3];
        if ( v11[4] >= v8 )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v13 )
          goto LABEL_6;
      }
      v10 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v12 );
LABEL_6:
    if ( v7 != v10 && v10[4] > v8 )
      v10 = (_QWORD *)(a1 + 16);
    v14 = (__int64 *)v10[5];
    v15 = v7;
    do
    {
      while ( 1 )
      {
        v16 = v9[2];
        v17 = v9[3];
        if ( v9[4] >= a3 )
          break;
        v9 = (_QWORD *)v9[3];
        if ( !v17 )
          goto LABEL_13;
      }
      v15 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v16 );
LABEL_13:
    if ( v15 != v7 && v15[4] <= a3 )
      v7 = v15;
  }
  else
  {
    v14 = *(__int64 **)(a1 + 56);
  }
  v22 = v7[5];
  if ( !a2 )
  {
    v25 = 0;
    v18 = v14[3];
    if ( v18 != v14[4] )
      goto LABEL_21;
LABEL_27:
    sub_D10B90(v14 + 2, v18, (__int64)v23, &v22);
    if ( !(_BYTE)v25 )
      goto LABEL_25;
    goto LABEL_28;
  }
  v23[0] = 6;
  v23[1] = 0;
  v24 = a2;
  if ( a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v23);
  LOBYTE(v25) = 1;
  v18 = v14[3];
  if ( v18 == v14[4] )
    goto LABEL_27;
LABEL_21:
  if ( v18 )
  {
    *(_BYTE *)(v18 + 24) = 0;
    if ( (_BYTE)v25 )
    {
      *(_QWORD *)v18 = 6;
      *(_QWORD *)(v18 + 8) = 0;
      v20 = v24;
      v21 = v24 == -4096;
      *(_QWORD *)(v18 + 16) = v24;
      if ( v20 != 0 && !v21 && v20 != -8192 )
        sub_BD6050((unsigned __int64 *)v18, v23[0] & 0xFFFFFFFFFFFFFFF8LL);
      *(_BYTE *)(v18 + 24) = 1;
    }
    *(_QWORD *)(v18 + 32) = v22;
    v18 = v14[3];
  }
  v14[3] = v18 + 40;
  if ( (_BYTE)v25 )
  {
LABEL_28:
    LOBYTE(v25) = 0;
    if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
      sub_BD60C0(v23);
  }
LABEL_25:
  result = v22;
  ++*(_DWORD *)(v22 + 40);
  return result;
}
