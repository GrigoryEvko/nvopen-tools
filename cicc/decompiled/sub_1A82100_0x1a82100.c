// Function: sub_1A82100
// Address: 0x1a82100
//
__int64 __fastcall sub_1A82100(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned __int8 v7; // r13
  __int64 i; // r14
  unsigned __int8 v9; // r15
  __int64 v10; // rdi
  unsigned __int8 v11; // al
  unsigned __int8 v14; // [rsp+17h] [rbp-39h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F96DB4 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_16;
  }
  v14 = 0;
  v5 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_4F96DB4)
                 + 160);
  *(_QWORD *)(a1 + 160) = v5;
  v6 = a2 + 72;
  while ( 1 )
  {
    v7 = 0;
    for ( i = *(_QWORD *)(a2 + 80); v6 != i; i = *(_QWORD *)(a2 + 80) )
    {
      v9 = 0;
      do
      {
        v10 = i;
        i = *(_QWORD *)(i + 8);
        v11 = sub_1AC9A80(v10 - 24, v5);
        if ( v11 )
          v9 = v11;
      }
      while ( v6 != i );
      if ( !v9 )
        break;
      v7 = v9;
    }
    if ( !v7 )
      break;
    sub_1AF0CE0(a2, 0, 0);
    v14 = v7;
    v5 = *(_QWORD *)(a1 + 160);
  }
  return v14;
}
