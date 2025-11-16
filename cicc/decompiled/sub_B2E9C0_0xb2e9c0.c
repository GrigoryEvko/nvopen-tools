// Function: sub_B2E9C0
// Address: 0xb2e9c0
//
__int16 __fastcall sub_B2E9C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 **v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx

  if ( a2 )
  {
    sub_B2E530(a1);
    v2 = *(_QWORD *)(a1 - 8);
    if ( *(_QWORD *)(v2 + 32) )
    {
      v3 = *(_QWORD *)(v2 + 40);
      **(_QWORD **)(v2 + 48) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = *(_QWORD *)(v2 + 48);
    }
    *(_QWORD *)(v2 + 32) = a2;
    v4 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v2 + 40) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = v2 + 40;
    *(_QWORD *)(v2 + 48) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v2 + 32;
  }
  else if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
  {
    v6 = *(_QWORD *)(a1 - 8);
    v7 = sub_B2BE50(a1);
    v8 = (__int64 **)sub_BCE3C0(v7, 0);
    v9 = sub_AC9EC0(v8);
    if ( *(_QWORD *)(v6 + 32) )
    {
      v10 = *(_QWORD *)(v6 + 40);
      **(_QWORD **)(v6 + 48) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v6 + 48);
    }
    *(_QWORD *)(v6 + 32) = v9;
    if ( v9 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v6 + 40) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v6 + 40;
      *(_QWORD *)(v6 + 48) = v9 + 16;
      *(_QWORD *)(v9 + 16) = v6 + 32;
    }
  }
  return sub_B2E700(a1, 1, a2 != 0);
}
