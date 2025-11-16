// Function: sub_D85880
// Address: 0xd85880
//
__int64 __fastcall sub_D85880(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax

  v2 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v2 + 8) == 14 )
  {
    if ( *(_DWORD *)(v2 + 8) >> 8 )
      return 0;
    else
      return sub_DD8400(*(_QWORD *)(a1 + 16), a2);
  }
  else
  {
    v4 = (__int64 *)sub_B2BE50(**(_QWORD **)(a1 + 16));
    v5 = sub_BCE3C0(v4, 0);
    v6 = *(_QWORD *)(a1 + 16);
    v7 = v5;
    v8 = sub_DD8400(v6, a2);
    return sub_DC5760(v6, v8, v7, 0);
  }
}
