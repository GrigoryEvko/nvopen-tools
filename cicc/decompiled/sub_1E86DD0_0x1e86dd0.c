// Function: sub_1E86DD0
// Address: 0x1e86dd0
//
unsigned __int64 __fastcall sub_1E86DD0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        signed int a6,
        int a7)
{
  unsigned __int64 v9; // r13
  __int64 *v11; // rcx
  __int64 v12; // rsi
  bool v13; // al
  char v14; // r9
  unsigned __int64 result; // rax
  __int64 v16; // rdi
  char v17; // [rsp+Fh] [rbp-41h]

  v9 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (__int64 *)sub_1DB3C70((__int64 *)a5, a4 & 0xFFFFFFFFFFFFFFF8LL);
  v12 = *(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8);
  if ( v11 == (__int64 *)v12
    || (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3) > *(_DWORD *)(v9 + 24) )
  {
    v13 = 1;
    v14 = 0;
  }
  else
  {
    v16 = v11[2];
    v14 = 0;
    if ( v9 != (v11[1] & 0xFFFFFFFFFFFFFFF8LL) || (v14 = 1, v13 = v16 == 0, (__int64 *)v12 != v11 + 3) )
      v13 = *(_QWORD *)(v16 + 8) == v9;
  }
  if ( !a7 && v13 )
  {
    v17 = v14;
    sub_1E86D40(a1, "No live segment at use", a2, a3, 0);
    sub_1E85C60(a5);
    if ( a6 < 0 )
      sub_1E85940(a1, a6);
    else
      sub_1E859F0(a1, a6);
    sub_1E85AA0(a4);
    v14 = v17;
  }
  result = (*(_BYTE *)(a2 + 3) & 0x40) != 0;
  if ( ((unsigned __int8)result & ((*(_BYTE *)(a2 + 3) >> 4) ^ 1)) != 0 && !v14 )
  {
    sub_1E86D40(a1, "Live range continues after kill flag", a2, a3, 0);
    sub_1E85C60(a5);
    if ( a6 < 0 )
      sub_1E85940(a1, a6);
    else
      sub_1E859F0(a1, a6);
    if ( a7 )
      sub_1E85CD0(a7);
    return (unsigned __int64)sub_1E85AA0(a4);
  }
  return result;
}
