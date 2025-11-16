// Function: sub_16BE010
// Address: 0x16be010
//
__int64 __fastcall sub_16BE010(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rax

  *(_QWORD *)a1 = &unk_49EF340;
  if ( *(_QWORD *)(a1 + 24) != *(_QWORD *)(a1 + 8) )
    sub_16E7BA0(a1);
  v1 = *(_QWORD *)(a1 + 40);
  if ( v1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    if ( !*(_DWORD *)(a1 + 32) || v2 )
    {
      v3 = *(_QWORD *)(a1 + 16) - v2;
    }
    else
    {
      v8 = sub_16E7720(a1);
      v1 = *(_QWORD *)(a1 + 40);
      v3 = v8;
    }
    v4 = *(_QWORD *)(v1 + 24);
    v5 = *(_QWORD *)(v1 + 8);
    if ( v3 )
    {
      if ( v4 != v5 )
        sub_16E7BA0(v1);
      v6 = sub_2207820(v3);
      sub_16E7A40(v1, v6, v3, 1);
    }
    else
    {
      if ( v4 != v5 )
        sub_16E7BA0(v1);
      sub_16E7A40(v1, 0, 0, 0);
    }
  }
  return sub_16E7960(a1);
}
