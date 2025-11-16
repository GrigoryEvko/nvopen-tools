// Function: sub_1ABB940
// Address: 0x1abb940
//
__int64 __fastcall sub_1ABB940(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 *i; // r15
  unsigned __int64 *v9; // rsi
  unsigned __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 *v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 72);
  result = *(_QWORD *)(a1 + 80);
  v4 = *v2;
  v13 = (__int64 *)result;
  if ( (__int64 *)result != v2 )
  {
    v5 = a2;
    v6 = *(_QWORD *)(*v2 + 56) + 72LL;
    v7 = a2 + 72;
    for ( i = v2 + 1; ; ++i )
    {
      v14 = v5;
      sub_15E0220(v6, v4);
      v9 = *(unsigned __int64 **)(v4 + 32);
      v10 = *(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      *v9 = v10 | *v9 & 7;
      *(_QWORD *)(v10 + 8) = v9;
      *(_QWORD *)(v4 + 24) &= 7uLL;
      *(_QWORD *)(v4 + 32) = 0;
      sub_15E01D0(v7, v4);
      v5 = v14;
      v11 = *(_QWORD *)(v14 + 72);
      v12 = *(_QWORD *)(v4 + 24) & 7LL;
      *(_QWORD *)(v4 + 32) = v7;
      v11 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v4 + 24) = v11 | v12;
      *(_QWORD *)(v11 + 8) = v4 + 24;
      result = *(_QWORD *)(v14 + 72) & 7LL;
      *(_QWORD *)(v14 + 72) = result | (v4 + 24);
      if ( v13 == i )
        break;
      v4 = *i;
      if ( !v4 )
      {
        sub_15E0220(v6, -24);
        BUG();
      }
    }
  }
  return result;
}
