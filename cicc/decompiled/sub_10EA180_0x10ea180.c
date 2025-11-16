// Function: sub_10EA180
// Address: 0x10ea180
//
__int64 __fastcall sub_10EA180(__int64 **a1)
{
  __int64 *v2; // r12
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( sub_CF91F0(**a1) )
    return sub_F207A0((__int64)a1[2], a1[1]);
  v2 = a1[2];
  v3 = (__int64 *)sub_BD5C60(**a1);
  v4 = sub_ACD6D0(v3);
  v5 = **a1;
  if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(v5 - 8);
  else
    v6 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
  v7 = *(_QWORD *)v6;
  if ( *(_QWORD *)v6 )
  {
    v8 = *(_QWORD *)(v6 + 8);
    **(_QWORD **)(v6 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v6 + 16);
  }
  *(_QWORD *)v6 = v4;
  if ( v4 )
  {
    v9 = *(_QWORD *)(v4 + 16);
    *(_QWORD *)(v6 + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v6 + 8;
    *(_QWORD *)(v6 + 16) = v4 + 16;
    *(_QWORD *)(v4 + 16) = v6;
  }
  if ( *(_BYTE *)v7 > 0x1Cu )
  {
    v10 = v2[5];
    v13[0] = v7;
    v11 = v10 + 2096;
    sub_10E8740(v11, v13);
    v12 = *(_QWORD *)(v7 + 16);
    if ( v12 )
    {
      if ( !*(_QWORD *)(v12 + 8) )
      {
        v13[0] = *(_QWORD *)(v12 + 24);
        sub_10E8740(v11, v13);
      }
    }
  }
  return 0;
}
