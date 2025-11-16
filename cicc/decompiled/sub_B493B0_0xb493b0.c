// Function: sub_B493B0
// Address: 0xb493b0
//
__int64 __fastcall sub_B493B0(_QWORD *a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r13d
  __int64 v3; // rax
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rax
  int v9; // eax
  unsigned __int64 v11; // rax
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = sub_A74710(a1 + 9, 0, 43);
  if ( (_BYTE)v1 )
    return 1;
  v2 = v1;
  v3 = *(a1 - 4);
  if ( v3 )
  {
    if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == a1[10] )
    {
      v12[0] = *(_QWORD *)(v3 + 120);
      if ( (unsigned __int8)sub_A74710(v12, 0, 43) )
        return 1;
    }
  }
  v4 = sub_A74620(a1 + 9);
  v5 = *(a1 - 4);
  if ( v5 )
  {
    if ( !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == a1[10] )
    {
      v12[0] = *(_QWORD *)(v5 + 120);
      v11 = sub_A74620(v12);
      if ( v4 < v11 )
        v4 = v11;
    }
  }
  if ( v4 )
  {
    v6 = a1[1];
    if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
      v6 = **(_QWORD **)(v6 + 16);
    v7 = *(_DWORD *)(v6 + 8);
    v8 = sub_B491C0((__int64)a1);
    LOBYTE(v9) = sub_B2F070(v8, v7 >> 8);
    return v9 ^ 1u;
  }
  return v2;
}
