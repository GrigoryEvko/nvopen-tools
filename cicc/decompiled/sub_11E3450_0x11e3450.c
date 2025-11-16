// Function: sub_11E3450
// Address: 0x11e3450
//
__int64 __fastcall sub_11E3450(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  unsigned int v10; // edx
  char *v11; // rax
  int v12; // esi
  size_t v13; // rdx
  unsigned int v14; // ecx
  unsigned __int64 *v15; // r15
  _BYTE *v16; // rax
  signed __int64 v17; // rax
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // r15
  _BYTE *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _BYTE *v26; // [rsp+0h] [rbp-A0h]
  unsigned int v27; // [rsp+Ch] [rbp-94h]
  size_t v28; // [rsp+10h] [rbp-90h]
  _BYTE *v30; // [rsp+28h] [rbp-78h] BYREF
  void *s; // [rsp+30h] [rbp-70h] BYREF
  size_t n; // [rsp+38h] [rbp-68h]
  _BYTE v33[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(a2 - 32 * v3);
  v5 = *(_QWORD *)(a2 + 32 * (1 - v3));
  v6 = *(_QWORD *)(a2 + 32 * (2 - v3));
  if ( *(_BYTE *)v6 != 17 )
    v6 = 0;
  v7 = *(_QWORD *)(a2 + 32 * (3 - v3));
  v8 = *(_QWORD *)(a2 + 16);
  if ( *(_BYTE *)v7 == 17 )
  {
    s = 0;
    v9 = a2;
    n = 0;
    if ( !v8 && v5 == v4 )
      return v4;
    if ( sub_AC30F0(v7) )
      return sub_AD6530(*(_QWORD *)(v9 + 8), a2);
    if ( (unsigned __int8)sub_98B0F0(v5, &s, 0) == 1 && v6 )
    {
      v10 = *(_DWORD *)(v6 + 32);
      v11 = *(char **)(v6 + 24);
      if ( v10 <= 0x40 )
      {
        v12 = 0;
        if ( v10 )
          v12 = (char)((__int64)((_QWORD)v11 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10));
      }
      else
      {
        v12 = *v11;
      }
      v13 = n;
      v14 = *(_DWORD *)(v7 + 32);
      v15 = *(unsigned __int64 **)(v7 + 24);
      if ( n )
      {
        v27 = *(_DWORD *)(v7 + 32);
        v28 = n;
        v26 = s;
        v16 = memchr(s, v12, n);
        v13 = v28;
        v14 = v27;
        if ( v16 )
        {
          v17 = v16 - v26;
          if ( v17 != -1 )
          {
            v18 = (unsigned __int64)v15;
            if ( v27 > 0x40 )
              v18 = *v15;
            v19 = v17 + 1;
            if ( v17 + 1 <= v18 )
              v18 = v17 + 1;
            v20 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v7 + 8), v18, 0);
            a2 = 238;
            v30 = v20;
            v21 = sub_B343C0(a3, 0xEEu, v4, 0x100u, v5, 0x100u, (__int64)v20, 0, 0, 0, 0, 0);
            if ( v21 && *(_BYTE *)v21 == 85 )
              *(_WORD *)(v21 + 2) = *(_WORD *)(v21 + 2) & 0xFFFC | *(_WORD *)(v9 + 2) & 3;
            v22 = *(_QWORD **)(v7 + 24);
            if ( *(_DWORD *)(v7 + 32) > 0x40u )
              v22 = (_QWORD *)*v22;
            if ( v19 <= (unsigned __int64)v22 )
            {
              v34 = 257;
              v23 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
              return sub_921130((unsigned int **)a3, v23, v4, &v30, 1, (__int64)v33, 3u);
            }
            return sub_AD6530(*(_QWORD *)(v9 + 8), a2);
          }
        }
      }
      if ( v14 > 0x40 )
        v15 = (unsigned __int64 *)*v15;
      if ( v13 >= (unsigned __int64)v15 )
      {
        a2 = 238;
        v24 = sub_B343C0(
                a3,
                0xEEu,
                v4,
                0x100u,
                v5,
                0x100u,
                *(_QWORD *)(v9 + 32 * (3LL - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF))),
                0,
                0,
                0,
                0,
                0);
        if ( v24 && *(_BYTE *)v24 == 85 )
          *(_WORD *)(v24 + 2) = *(_WORD *)(v24 + 2) & 0xFFFC | *(_WORD *)(v9 + 2) & 3;
        return sub_AD6530(*(_QWORD *)(v9 + 8), a2);
      }
    }
    return 0;
  }
  if ( v8 || v5 != v4 )
    return 0;
  return v4;
}
