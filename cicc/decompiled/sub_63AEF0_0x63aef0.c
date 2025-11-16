// Function: sub_63AEF0
// Address: 0x63aef0
//
__int64 __fastcall sub_63AEF0(__int64 a1, __int64 a2)
{
  char v4; // al
  __int64 v5; // rdx
  __int64 v6; // rsi
  bool v7; // cc
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // [rsp+8h] [rbp-1F8h] BYREF
  _QWORD v20[61]; // [rsp+10h] [rbp-1F0h] BYREF

  memset(v20, 0, 0x1D8u);
  v20[19] = v20;
  if ( dword_4F077BC )
  {
    v4 = BYTE2(v20[22]);
    if ( qword_4F077A8 <= 0x9F5Fu )
      v4 = BYTE2(v20[22]) | 1;
  }
  else
  {
    v4 = BYTE2(v20[22]);
  }
  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(a1 + 120);
  v7 = *(_BYTE *)(a1 + 136) <= 2u;
  BYTE2(v20[22]) = v4 | 2;
  v20[0] = v5;
  v8 = *(_QWORD *)(a1 + 64);
  v20[35] = v6;
  v20[3] = v8;
  v20[6] = v8;
  v20[36] = v6;
  if ( v7 )
    LOBYTE(v20[22]) |= 2u;
  v9 = a2;
  result = sub_694AA0(a2, v6, 1, 1, &v20[17]);
  v15 = v20[18];
  if ( (v20[22] & 0x200LL) != 0 && !v20[17] )
  {
    if ( v20[18] )
      goto LABEL_8;
    v19 = sub_724DC0(v9, 0, v11, v12, v13, v14);
    sub_72C970(v19);
    result = sub_724E50(&v19, 0, v16, v17, v18);
    v20[18] = 0;
    v20[17] = result;
  }
  else if ( v20[18] )
  {
LABEL_8:
    v19 = 0;
    return sub_630370(a1, v15, &v19, a1 + 64, 0, 0);
  }
  if ( *(_BYTE *)(a1 + 136) > 2u )
  {
    result = sub_725A70(2);
    v20[18] = result;
    *(_QWORD *)(result + 56) = v20[17];
    v15 = v20[18];
    v20[17] = 0;
    if ( v20[18] )
      goto LABEL_8;
  }
  return result;
}
