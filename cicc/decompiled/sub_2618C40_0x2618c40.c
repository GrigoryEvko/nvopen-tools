// Function: sub_2618C40
// Address: 0x2618c40
//
bool __fastcall sub_2618C40(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  bool v6; // zf

  v1 = *(_DWORD *)(a1 + 44);
  if ( v1 != -1 )
    return v1 != 0;
  v3 = sub_BA91D0(*(_QWORD *)a1, "branch-target-enforcement", 0x19u);
  if ( v3 && (v4 = *(_QWORD *)(v3 + 136)) != 0 )
  {
    if ( *(_DWORD *)(v4 + 32) <= 0x40u )
      v5 = *(_QWORD *)(v4 + 24);
    else
      v5 = **(_QWORD **)(v4 + 24);
    v6 = v5 == 0;
    *(_DWORD *)(a1 + 44) = !v6;
    return v5 != 0;
  }
  else
  {
    *(_DWORD *)(a1 + 44) = 0;
    return 0;
  }
}
