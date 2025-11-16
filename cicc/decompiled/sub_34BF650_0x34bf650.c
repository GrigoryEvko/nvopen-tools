// Function: sub_34BF650
// Address: 0x34bf650
//
__int64 __fastcall sub_34BF650(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v10; // rbx
  bool v12; // r9
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rsi
  bool v16; // [rsp+Fh] [rbp-41h]
  __int64 *v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  result = a1[1];
  v8 = *a1;
  if ( a2 == *(_DWORD *)(result - 24) )
  {
    v10 = result - 24;
    v12 = a3 != 0;
    while ( 1 )
    {
      v13 = *(_QWORD *)(v10 + 8);
      if ( v13 != a4 && v12 )
      {
        v16 = v12;
        v17 = a5;
        sub_34BEAF0(v13, a3, a1[17], a5);
        v12 = v16;
        a5 = v17;
      }
      if ( v8 == v10 )
        break;
      if ( a2 != *(_DWORD *)(v10 - 24) )
      {
        result = a1[1];
        goto LABEL_11;
      }
      v10 -= 24;
    }
    result = a1[1];
    if ( a2 != *(_DWORD *)v10 )
      v10 += 24;
  }
  else
  {
    v10 = a1[1];
  }
LABEL_11:
  if ( v10 != result )
  {
    v14 = v10;
    do
    {
      v15 = *(_QWORD *)(v14 + 16);
      if ( v15 )
      {
        v18 = result;
        sub_B91220(v14 + 16, v15);
        result = v18;
      }
      v14 += 24;
    }
    while ( v14 != result );
    a1[1] = v10;
  }
  return result;
}
