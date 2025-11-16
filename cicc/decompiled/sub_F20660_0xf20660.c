// Function: sub_F20660
// Address: 0xf20660
//
__int64 __fastcall sub_F20660(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8) + 32LL * a3;
  else
    v4 = a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = *(_QWORD *)v4;
  if ( *(_QWORD *)v4 )
  {
    v6 = *(_QWORD *)(v4 + 8);
    **(_QWORD **)(v4 + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v4 + 16);
  }
  *(_QWORD *)v4 = a4;
  if ( a4 )
  {
    v7 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v4 + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = v4 + 8;
    *(_QWORD *)(v4 + 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v4;
  }
  if ( *(_BYTE *)v5 > 0x1Cu )
  {
    v8 = *(_QWORD *)(a1 + 40);
    v12[0] = v5;
    v9 = v8 + 2096;
    sub_F200C0(v9, v12);
    v10 = *(_QWORD *)(v5 + 16);
    if ( v10 )
    {
      if ( !*(_QWORD *)(v10 + 8) )
      {
        v12[0] = *(_QWORD *)(v10 + 24);
        sub_F200C0(v9, v12);
      }
    }
  }
  return a2;
}
