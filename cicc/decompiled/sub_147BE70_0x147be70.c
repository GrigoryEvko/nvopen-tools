// Function: sub_147BE70
// Address: 0x147be70
//
__int64 __fastcall sub_147BE70(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v6; // ax
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 *v16; // rcx
  __int64 *v17; // rbx
  __int64 v18; // rsi
  __int64 *v19; // [rsp+18h] [rbp-78h]
  __int64 v20; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v21[2]; // [rsp+30h] [rbp-60h] BYREF
  _BYTE v22[80]; // [rsp+40h] [rbp-50h] BYREF

  while ( 1 )
  {
    a3 = sub_1456E10(a1, a3);
    v6 = *(_WORD *)(a2 + 24);
    if ( !v6 )
      break;
    if ( v6 != 1 )
      goto LABEL_5;
    a2 = *(_QWORD *)(a2 + 32);
    v12 = sub_1456040(a2);
    v13 = sub_1456C90(a1, v12);
    if ( v13 >= sub_1456C90(a1, a3) )
      return sub_1483C80(a1, a2, a3);
  }
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_DWORD *)(v7 + 32);
  v9 = *(_QWORD *)(v7 + 24);
  if ( v8 > 0x40 )
    v9 = *(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6));
  if ( (v9 & (1LL << ((unsigned __int8)v8 - 1))) != 0 )
    return sub_147B0D0(a1, a2, a3, 0);
LABEL_5:
  v10 = sub_14747F0(a1, a2, a3, 0);
  if ( *(_WORD *)(v10 + 24) == 2 )
  {
    v14 = sub_147B0D0(a1, a2, a3, 0);
    if ( *(_WORD *)(v14 + 24) == 3 )
    {
      v15 = *(_WORD *)(a2 + 24);
      if ( v15 == 7 )
      {
        v16 = *(__int64 **)(a2 + 32);
        v21[0] = (__int64 *)v22;
        v21[1] = (__int64 *)0x400000000LL;
        v19 = &v16[*(_QWORD *)(a2 + 40)];
        if ( v16 != v19 )
        {
          v17 = v16;
          do
          {
            v18 = *v17++;
            v20 = sub_147BE70(a1, v18, a3);
            sub_1458920((__int64)v21, &v20);
          }
          while ( v19 != v17 );
        }
        v10 = sub_14785F0(a1, v21, *(_QWORD *)(a2 + 48), 1u);
        if ( (_BYTE *)v21[0] != v22 )
          _libc_free((unsigned __int64)v21[0]);
      }
      else if ( v15 == 9 )
      {
        return v14;
      }
    }
    else
    {
      return v14;
    }
  }
  return v10;
}
