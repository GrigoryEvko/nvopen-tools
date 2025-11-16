// Function: sub_39B30C0
// Address: 0x39b30c0
//
__int64 *__fastcall sub_39B30C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 v9; // rax

  v2 = sub_22077B0(0x40u);
  v3 = v2;
  if ( !v2 )
    goto LABEL_8;
  *(_DWORD *)(v2 + 32) = 1;
  v4 = *(_DWORD *)(a2 + 32);
  *(_QWORD *)(v2 + 24) = 0;
  *(_QWORD *)(v2 + 16) = 0;
  *(_QWORD *)(v2 + 8) = 0;
  *(_QWORD *)(v2 + 48) = 0;
  *(_QWORD *)(v2 + 40) = a2;
  *(_QWORD *)v2 = &unk_49EF340;
  v5 = *(_QWORD *)(a2 + 8);
  if ( !v4 || v5 )
  {
    v6 = *(_QWORD *)(a2 + 16) - v5;
    if ( v6 )
    {
LABEL_13:
      v9 = sub_2207820(v6);
      sub_16E7A40(v3, v9, v6, 1);
      goto LABEL_5;
    }
  }
  else
  {
    v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 72LL))(a2);
    if ( v6 )
    {
      if ( *(_QWORD *)(v3 + 24) != *(_QWORD *)(v3 + 8) )
        sub_16E7BA0((__int64 *)v3);
      goto LABEL_13;
    }
    if ( *(_QWORD *)(v3 + 8) != *(_QWORD *)(v3 + 24) )
      sub_16E7BA0((__int64 *)v3);
  }
  sub_16E7A40(v3, 0, 0, 0);
LABEL_5:
  v7 = *(_QWORD *)(v3 + 40);
  if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v7 + 8) )
    sub_16E7BA0(*(__int64 **)(v3 + 40));
  sub_16E7A40(v7, 0, 0, 0);
  *(_QWORD *)(v3 + 56) = 0;
LABEL_8:
  *a1 = v3;
  return a1;
}
