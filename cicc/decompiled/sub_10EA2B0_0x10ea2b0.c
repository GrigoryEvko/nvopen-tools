// Function: sub_10EA2B0
// Address: 0x10ea2b0
//
void __fastcall sub_10EA2B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)a2;
  if ( *(_QWORD *)a2 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    **(_QWORD **)(a2 + 16) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(a2 + 16);
  }
  *(_QWORD *)a2 = a3;
  if ( a3 )
  {
    v5 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a2 + 8) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = a2 + 8;
    *(_QWORD *)(a2 + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a2;
  }
  if ( *(_BYTE *)v3 > 0x1Cu )
  {
    v6 = *(_QWORD *)(a1 + 40);
    v9[0] = v3;
    v7 = v6 + 2096;
    sub_10E8740(v7, v9);
    v8 = *(_QWORD *)(v3 + 16);
    if ( v8 )
    {
      if ( !*(_QWORD *)(v8 + 8) )
      {
        v9[0] = *(_QWORD *)(v8 + 24);
        sub_10E8740(v7, v9);
      }
    }
  }
}
