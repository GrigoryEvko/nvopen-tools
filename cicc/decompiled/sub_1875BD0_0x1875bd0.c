// Function: sub_1875BD0
// Address: 0x1875bd0
//
__int64 __fastcall sub_1875BD0(__int64 ***a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  __int64 **v8; // rdx
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  __int64 v14; // r15
  __int64 **v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r12
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26[8]; // [rsp+0h] [rbp-40h] BYREF

  v8 = *a1;
  if ( (unsigned int)(*((_DWORD *)*a1 + 6) - 31) <= 1 && *((_DWORD *)v8 + 8) == 2 )
  {
    v11 = sub_1875AC0(a1[1], a2, a3);
    v14 = sub_1649C60(v11);
    if ( *(_BYTE *)(a6 + 8) == 11 )
      v11 = sub_15A4180(v11, (__int64 **)a6, 0);
    if ( !sub_1626AA0(v14, 21) )
    {
      v15 = *a1;
      v16 = (__int64)(*a1)[12];
      if ( a5 == *(_DWORD *)(v16 + 8) >> 8 )
      {
        v24 = sub_159C470(v16, -1, 0);
        v25 = sub_1624210(v24);
        v19 = (__int64)v15[12];
        v21 = -1;
        v20 = (__int64)v25;
      }
      else
      {
        v17 = sub_159C470(v16, 0, 0);
        v18 = sub_1624210(v17);
        v19 = (__int64)v15[12];
        v20 = (__int64)v18;
        v21 = 1LL << a5;
      }
      v22 = sub_159C470(v19, v21, 0);
      v26[0] = v20;
      v26[1] = (__int64)sub_1624210(v22);
      v23 = sub_1627350((__int64 *)**v15, v26, (__int64 *)2, 0, 1);
      sub_16270B0(v14, 0x15u, v23);
    }
    return v11;
  }
  v9 = a6;
  if ( *(_BYTE *)(a6 + 8) != 11 )
    v9 = (__int64)v8[11];
  v10 = sub_15A0680(v9, a4, 0);
  v11 = v10;
  if ( *(_BYTE *)(a6 + 8) == 11 )
    return v11;
  return sub_15A3BA0(v10, (__int64 **)a6, 0);
}
