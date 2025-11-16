// Function: sub_154B550
// Address: 0x154b550
//
__int64 __fastcall sub_154B550(__int64 a1, __int64 a2)
{
  int v2; // eax
  bool v3; // zf
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // rax

  *(_DWORD *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)a1 = &unk_49EF340;
  v2 = *(_DWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = a2;
  v3 = v2 == 0;
  v4 = *(_QWORD *)(a2 + 8);
  if ( v3 || v4 )
  {
    v5 = *(_QWORD *)(a2 + 16) - v4;
    if ( !v5 )
    {
LABEL_4:
      sub_16E7A40(a1, 0, 0, 0);
      goto LABEL_5;
    }
  }
  else
  {
    v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 72LL))(a2);
    if ( !v5 )
    {
      if ( *(_QWORD *)(a1 + 24) != *(_QWORD *)(a1 + 8) )
        sub_16E7BA0(a1);
      goto LABEL_4;
    }
    if ( *(_QWORD *)(a1 + 24) != *(_QWORD *)(a1 + 8) )
      sub_16E7BA0(a1);
  }
  v8 = sub_2207820(v5);
  sub_16E7A40(a1, v8, v5, 1);
LABEL_5:
  v6 = *(_QWORD *)(a1 + 40);
  if ( *(_QWORD *)(v6 + 24) != *(_QWORD *)(v6 + 8) )
    sub_16E7BA0(*(_QWORD *)(a1 + 40));
  result = sub_16E7A40(v6, 0, 0, 0);
  *(_QWORD *)(a1 + 56) = 0;
  return result;
}
