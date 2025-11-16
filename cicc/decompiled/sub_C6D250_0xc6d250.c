// Function: sub_C6D250
// Address: 0xc6d250
//
__int64 __fastcall sub_C6D250(__int64 a1, unsigned __int16 *a2, unsigned __int64 a3)
{
  __int64 result; // rax
  unsigned __int16 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int16 *v8; // r15
  unsigned __int16 *v9; // r14
  __int16 *v10; // r12
  unsigned __int16 *v11; // rdi
  __int16 *v12; // rax
  unsigned __int16 *v13; // r14
  unsigned __int16 *v14; // rsi
  __int64 v15; // [rsp+8h] [rbp-38h]

  result = 0x333333333333333LL;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( a3 > 0x333333333333333LL )
    sub_4262D8((__int64)"vector::reserve");
  if ( a3 )
  {
    v5 = a2;
    v15 = 20 * a3;
    v6 = sub_22077B0(40 * a3);
    v8 = *(unsigned __int16 **)(a1 + 8);
    v9 = *(unsigned __int16 **)a1;
    v10 = (__int16 *)v6;
    if ( v8 != *(unsigned __int16 **)a1 )
    {
      do
      {
        v11 = v9;
        v9 += 20;
        sub_C6BC50(v11);
      }
      while ( v8 != v9 );
      v9 = *(unsigned __int16 **)a1;
    }
    if ( v9 )
      j_j___libc_free_0(v9, *(_QWORD *)(a1 + 16) - (_QWORD)v9);
    *(_QWORD *)a1 = v10;
    *(_QWORD *)(a1 + 8) = v10;
    v12 = &v10[v15];
    v13 = &a2[v15];
    for ( *(_QWORD *)(a1 + 16) = &v10[v15]; ; v12 = *(__int16 **)(a1 + 16) )
    {
      if ( v12 == v10 )
      {
        sub_C6D0A0((__int16 **)a1, v10, v7);
        v10 = (__int16 *)(*(_QWORD *)(a1 + 8) - 40LL);
      }
      else
      {
        if ( v10 )
        {
          *v10 = 0;
          v10 = *(__int16 **)(a1 + 8);
        }
        *(_QWORD *)(a1 + 8) = v10 + 20;
      }
      v14 = v5;
      v5 += 20;
      result = (__int64)sub_C6A4F0((__int64)v10, v14);
      if ( v13 == v5 )
        break;
      v10 = *(__int16 **)(a1 + 8);
    }
  }
  return result;
}
