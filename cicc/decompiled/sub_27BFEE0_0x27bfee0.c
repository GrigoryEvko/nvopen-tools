// Function: sub_27BFEE0
// Address: 0x27bfee0
//
unsigned __int64 __fastcall sub_27BFEE0(__int64 a1, unsigned __int8 *a2)
{
  int v2; // r12d
  unsigned __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v8; // rax
  unsigned __int8 *v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // rax
  int v13; // edx
  int v14; // r15d
  int v15; // edx
  bool v16; // cc
  __int64 v17; // rsi
  bool v18; // r12
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  unsigned __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = *a2;
  result = (unsigned int)(v2 - 68);
  if ( (unsigned __int8)(v2 - 68) <= 1u )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v5 = *((_QWORD *)a2 + 1);
    v20 = *(_QWORD *)(a1 + 24);
    v21 = sub_D97050(v4, v5);
    v8 = sub_B43CC0((__int64)a2);
    v9 = *(unsigned __int8 **)(v8 + 32);
    v10 = *(_QWORD *)(v8 + 40);
    v22[0] = v21;
    v11 = (__int64)&v9[v10];
    result = (unsigned __int64)sub_27BFE20(v9, v11, (__int64 *)v22);
    if ( v11 != result )
    {
      result = sub_D97050(v4, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL));
      if ( v21 > result )
      {
        if ( !v20 )
          goto LABEL_9;
        v12 = sub_DFD800(v20, 0xDu, *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL), 0, 0, 0, 0, 0, 0, 0);
        v14 = v13;
        v19 = v12;
        result = sub_DFD800(v20, 0xDu, v5, 0, 0, 0, 0, 0, 0, 0);
        v16 = v14 < v15;
        if ( v14 == v15 )
          v16 = v19 < (__int64)result;
        if ( !v16 )
        {
LABEL_9:
          v17 = *(_QWORD *)(a1 + 48);
          v18 = (_BYTE)v2 == 69;
          if ( v17 && (result = sub_D97050(v4, v17), v21 <= result) )
          {
            *(_BYTE *)(a1 + 56) |= v18;
          }
          else
          {
            result = sub_D97090(v4, v5);
            *(_BYTE *)(a1 + 56) = v18;
            *(_QWORD *)(a1 + 48) = result;
          }
        }
      }
    }
  }
  return result;
}
