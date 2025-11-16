// Function: sub_30278D0
// Address: 0x30278d0
//
void __fastcall sub_30278D0(__int64 a1, unsigned int a2, __int64 a3)
{
  _BYTE *v4; // r13
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  _BYTE *v9; // rax

  v4 = *(_BYTE **)(*(_QWORD *)(a1 + 64) + 8LL * a2);
  v5 = *(_QWORD *)(a1 + 112);
  v6 = *(_QWORD *)(v5 + 8LL * a2);
  if ( *v4 > 3u )
  {
    if ( *(_BYTE *)v6 != 5 )
      BUG();
    v9 = (_BYTE *)sub_30270A0(*(_QWORD *)(a1 + 168), *(unsigned __int8 **)(v5 + 8LL * a2), 0);
    sub_30275E0(*(_QWORD *)(a1 + 168), v9, a3);
  }
  else
  {
    v7 = sub_31DB510(*(_QWORD *)(a1 + 168), v4);
    v8 = *(_QWORD *)(v6 + 8);
    if ( *(_BYTE *)(v8 + 8) == 14 && !(*(_DWORD *)(v8 + 8) >> 8) && *(_BYTE *)(a1 + 176) && *v4 )
    {
      sub_904010(a3, "generic(");
      sub_EA12C0(v7, a3, *(_BYTE **)(*(_QWORD *)(a1 + 168) + 208LL));
      sub_904010(a3, ")");
    }
    else
    {
      sub_EA12C0(v7, a3, *(_BYTE **)(*(_QWORD *)(a1 + 168) + 208LL));
    }
  }
}
