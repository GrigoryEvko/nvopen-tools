// Function: sub_3373D80
// Address: 0x3373d80
//
__int64 __fastcall sub_3373D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v4; // rsi
  __int64 v5; // rdi
  _QWORD *v7; // rsi
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  unsigned int v10; // edx
  unsigned int v11; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v12; // [rsp-8h] [rbp-8h]

  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 960) + 32LL);
  if ( v5 )
    return sub_FF0430(v5, v4, *(_QWORD *)(a3 + 16));
  v12 = v3;
  v7 = (_QWORD *)(v4 + 48);
  v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v8 == v7 )
    goto LABEL_9;
  if ( !v8 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
  {
LABEL_9:
    v10 = 1;
  }
  else
  {
    v9 = sub_B46E30(v8 - 24);
    v10 = 1;
    if ( v9 )
      v10 = v9;
  }
  sub_F02DB0(&v11, 1u, v10);
  return v11;
}
