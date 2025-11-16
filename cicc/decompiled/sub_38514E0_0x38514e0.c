// Function: sub_38514E0
// Address: 0x38514e0
//
__int64 __fastcall sub_38514E0(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rbx
  _QWORD *v3; // rdi
  unsigned int v4; // r12d
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD v25[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = *a1;
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = (_QWORD *)(v2 + 56);
  if ( (v1 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v3, -1, 36) )
    {
      if ( *(char *)(v2 + 23) < 0 )
      {
        v6 = sub_1648A40(v2);
        v8 = v6 + v7;
        v9 = 0;
        if ( *(char *)(v2 + 23) < 0 )
          v9 = sub_1648A40(v2);
        if ( (unsigned int)((v8 - v9) >> 4) )
          goto LABEL_9;
      }
      v15 = *(_QWORD *)(v2 - 24);
      if ( *(_BYTE *)(v15 + 16) || (v25[0] = *(_QWORD *)(v15 + 112), !(unsigned __int8)sub_1560260(v25, -1, 36)) )
      {
LABEL_9:
        v4 = sub_1560260((_QWORD *)(v2 + 56), -1, 37);
        if ( !(_BYTE)v4 )
        {
          if ( *(char *)(v2 + 23) < 0 )
          {
            v10 = sub_1648A40(v2);
            v12 = v10 + v11;
            v13 = *(char *)(v2 + 23) >= 0 ? 0LL : sub_1648A40(v2);
            if ( v13 != v12 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v13 + 8LL) <= 1u )
              {
                v13 += 16;
                if ( v12 == v13 )
                  goto LABEL_16;
              }
              return v4;
            }
          }
LABEL_16:
          v14 = *(_QWORD *)(v2 - 24);
          if ( !*(_BYTE *)(v14 + 16) )
          {
LABEL_17:
            v25[0] = *(_QWORD *)(v14 + 112);
            return (unsigned int)sub_1560260(v25, -1, 37);
          }
          return 0;
        }
      }
    }
    return 1;
  }
  if ( (unsigned __int8)sub_1560260(v3, -1, 36) )
    return 1;
  if ( *(char *)(v2 + 23) >= 0 )
    goto LABEL_42;
  v16 = sub_1648A40(v2);
  v18 = v16 + v17;
  v19 = 0;
  if ( *(char *)(v2 + 23) < 0 )
    v19 = sub_1648A40(v2);
  if ( !(unsigned int)((v18 - v19) >> 4) )
  {
LABEL_42:
    v24 = *(_QWORD *)(v2 - 72);
    if ( !*(_BYTE *)(v24 + 16) )
    {
      v25[0] = *(_QWORD *)(v24 + 112);
      if ( (unsigned __int8)sub_1560260(v25, -1, 36) )
        return 1;
    }
  }
  v4 = sub_1560260((_QWORD *)(v2 + 56), -1, 37);
  if ( (_BYTE)v4 )
    return 1;
  if ( *(char *)(v2 + 23) >= 0
    || ((v20 = sub_1648A40(v2), v22 = v20 + v21, *(char *)(v2 + 23) >= 0) ? (v23 = 0) : (v23 = sub_1648A40(v2)),
        v23 == v22) )
  {
LABEL_33:
    v14 = *(_QWORD *)(v2 - 72);
    if ( !*(_BYTE *)(v14 + 16) )
      goto LABEL_17;
    return 0;
  }
  while ( *(_DWORD *)(*(_QWORD *)v23 + 8LL) <= 1u )
  {
    v23 += 16;
    if ( v22 == v23 )
      goto LABEL_33;
  }
  return v4;
}
