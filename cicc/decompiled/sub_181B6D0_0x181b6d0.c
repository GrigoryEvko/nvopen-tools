// Function: sub_181B6D0
// Address: 0x181b6d0
//
_QWORD *__fastcall sub_181B6D0(__int64 **a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r10
  _QWORD *v4; // rax
  __int64 v5; // rbx
  _QWORD *v6; // r15
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r12
  _QWORD *result; // rax
  __int64 *v20; // rbx
  unsigned __int64 v21; // rax
  __int64 v22; // r12
  unsigned __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  _BYTE *v25; // [rsp+18h] [rbp-58h]
  __int64 v26[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v27; // [rsp+30h] [rbp-40h]

  v25 = (_BYTE *)sub_1819D40(*a1, *(_QWORD *)(a2 - 72));
  v2 = sub_1819D40(*a1, *(_QWORD *)(a2 - 48));
  v3 = sub_1819D40(*a1, *(_QWORD *)(a2 - 24));
  if ( *(_BYTE *)(**(_QWORD **)(a2 - 72) + 8LL) == 16 )
  {
    v20 = *a1;
    v21 = sub_181A560(*a1, (_BYTE *)v2, v3, a2);
    v26[0] = a2;
    v22 = sub_181A560(v20, v25, v21, a2);
    result = sub_176FB00((__int64)(v20 + 16), v26);
    result[1] = v22;
  }
  else
  {
    v24 = *(_QWORD *)(a2 - 72);
    if ( v2 != v3 )
    {
      v23 = v3;
      v27 = 257;
      v4 = sub_1648A60(56, 3u);
      v5 = (__int64)v4;
      if ( v4 )
      {
        v6 = v4 - 9;
        sub_15F1EA0((__int64)v4, *(_QWORD *)v2, 55, (__int64)(v4 - 9), 3, a2);
        if ( *(_QWORD *)(v5 - 72) )
        {
          v7 = *(_QWORD *)(v5 - 64);
          v8 = *(_QWORD *)(v5 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v8 = v7;
          if ( v7 )
            *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
        }
        *(_QWORD *)(v5 - 72) = v24;
        v9 = *(_QWORD *)(v24 + 8);
        *(_QWORD *)(v5 - 64) = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = (v5 - 64) | *(_QWORD *)(v9 + 16) & 3LL;
        *(_QWORD *)(v5 - 56) = (v24 + 8) | *(_QWORD *)(v5 - 56) & 3LL;
        *(_QWORD *)(v24 + 8) = v6;
        if ( *(_QWORD *)(v5 - 48) )
        {
          v10 = *(_QWORD *)(v5 - 40);
          v11 = *(_QWORD *)(v5 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v11 = v10;
          if ( v10 )
            *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
        }
        *(_QWORD *)(v5 - 48) = v2;
        v12 = *(_QWORD *)(v2 + 8);
        *(_QWORD *)(v5 - 40) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = (v5 - 40) | *(_QWORD *)(v12 + 16) & 3LL;
        *(_QWORD *)(v5 - 32) = (v2 + 8) | *(_QWORD *)(v5 - 32) & 3LL;
        *(_QWORD *)(v2 + 8) = v5 - 48;
        if ( *(_QWORD *)(v5 - 24) )
        {
          v13 = *(_QWORD *)(v5 - 16);
          v14 = *(_QWORD *)(v5 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v14 = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
        }
        *(_QWORD *)(v5 - 24) = v23;
        if ( v23 )
        {
          v15 = *(_QWORD *)(v23 + 8);
          *(_QWORD *)(v5 - 16) = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 16) = (v5 - 16) | *(_QWORD *)(v15 + 16) & 3LL;
          *(_QWORD *)(v5 - 8) = (v23 + 8) | *(_QWORD *)(v5 - 8) & 3LL;
          *(_QWORD *)(v23 + 8) = v5 - 24;
        }
        sub_164B780(v5, v26);
      }
      v3 = v5;
    }
    v16 = *a1;
    v17 = sub_181A560(*a1, v25, v3, a2);
    v26[0] = a2;
    v18 = v17;
    result = sub_176FB00((__int64)(v16 + 16), v26);
    result[1] = v18;
  }
  return result;
}
