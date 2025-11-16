// Function: sub_2784800
// Address: 0x2784800
//
__int64 __fastcall sub_2784800(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned int v6; // r14d
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_9:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F86530 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_9;
  }
  v6 = 0;
  v7 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_4F86530)
                 + 176);
  for ( *(_QWORD *)(a1 + 176) = v7; ; v7 = *(_QWORD *)(a1 + 176) )
  {
    v8 = sub_2784530(a2, v7);
    if ( !(_BYTE)v8 )
      break;
    v6 = v8;
    sub_F62E00(a2, 0, 0, v9, v10, v11);
  }
  return v6;
}
