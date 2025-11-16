// Function: sub_14AB140
// Address: 0x14ab140
//
__int64 __fastcall sub_14AB140(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // rbx
  __int64 *v6; // rax
  char v7; // r12
  __int64 v8; // rsi
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r12
  __int64 v30; // rax
  unsigned int v31; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v32[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a1;
  v5 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (__int64 *)(v5 - 72);
  v7 = (v4 >> 2) & 1;
  if ( v7 )
    v6 = (__int64 *)(v5 - 24);
  v8 = *v6;
  if ( *(_BYTE *)(*v6 + 16) )
    return 0;
  if ( (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
    return *(unsigned int *)(v8 + 36);
  if ( !a2 || (*(_BYTE *)(v8 + 32) & 0xFu) - 7 <= 1 || !sub_149CB50(*a2, v8, &v31) )
    return 0;
  v10 = v5 + 56;
  if ( v7 )
  {
    if ( !(unsigned __int8)sub_1560260(v10, 0xFFFFFFFFLL, 36) )
    {
      if ( *(char *)(v5 + 23) < 0 )
      {
        v11 = sub_1648A40(v5);
        v13 = v11 + v12;
        v14 = 0;
        if ( *(char *)(v5 + 23) < 0 )
          v14 = sub_1648A40(v5);
        if ( (unsigned int)((v13 - v14) >> 4) )
          goto LABEL_50;
      }
      v15 = *(_QWORD *)(v5 - 24);
      if ( *(_BYTE *)(v15 + 16)
        || (v32[0] = *(_QWORD *)(v15 + 112), !(unsigned __int8)sub_1560260(v32, 0xFFFFFFFFLL, 36)) )
      {
LABEL_50:
        if ( !(unsigned __int8)sub_1560260(v5 + 56, 0xFFFFFFFFLL, 37) )
        {
          if ( *(char *)(v5 + 23) < 0 )
          {
            v16 = sub_1648A40(v5);
            v18 = v16 + v17;
            v19 = *(char *)(v5 + 23) >= 0 ? 0LL : sub_1648A40(v5);
            if ( v19 != v18 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v19 + 8LL) <= 1u )
              {
                v19 += 16;
                if ( v18 == v19 )
                  goto LABEL_25;
              }
              return 0;
            }
          }
LABEL_25:
          v20 = *(_QWORD *)(v5 - 24);
          if ( *(_BYTE *)(v20 + 16) )
            return 0;
          goto LABEL_26;
        }
      }
    }
  }
  else if ( !(unsigned __int8)sub_1560260(v10, 0xFFFFFFFFLL, 36) )
  {
    if ( *(char *)(v5 + 23) < 0 )
    {
      v21 = sub_1648A40(v5);
      v23 = v21 + v22;
      v24 = 0;
      if ( *(char *)(v5 + 23) < 0 )
        v24 = sub_1648A40(v5);
      if ( (unsigned int)((v23 - v24) >> 4) )
        goto LABEL_51;
    }
    v25 = *(_QWORD *)(v5 - 72);
    if ( *(_BYTE *)(v25 + 16) || (v32[0] = *(_QWORD *)(v25 + 112), !(unsigned __int8)sub_1560260(v32, 0xFFFFFFFFLL, 36)) )
    {
LABEL_51:
      if ( !(unsigned __int8)sub_1560260(v5 + 56, 0xFFFFFFFFLL, 37) )
      {
        if ( *(char *)(v5 + 23) < 0 )
        {
          v27 = sub_1648A40(v5);
          v29 = v27 + v28;
          v30 = *(char *)(v5 + 23) >= 0 ? 0LL : sub_1648A40(v5);
          if ( v30 != v29 )
          {
            while ( *(_DWORD *)(*(_QWORD *)v30 + 8LL) <= 1u )
            {
              v30 += 16;
              if ( v29 == v30 )
                goto LABEL_44;
            }
            return 0;
          }
        }
LABEL_44:
        v20 = *(_QWORD *)(v5 - 72);
        if ( *(_BYTE *)(v20 + 16) )
          return 0;
LABEL_26:
        v32[0] = *(_QWORD *)(v20 + 112);
        if ( (unsigned __int8)sub_1560260(v32, 0xFFFFFFFFLL, 37) )
          goto LABEL_35;
        return 0;
      }
    }
  }
LABEL_35:
  v26 = v31 - 155;
  if ( (unsigned int)v26 > 0xFA )
    return 0;
  return byte_428F8C0[v26];
}
