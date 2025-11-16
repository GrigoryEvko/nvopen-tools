// Function: sub_27DDE90
// Address: 0x27dde90
//
_BOOL8 __fastcall sub_27DDE90(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // r14
  _BOOL8 result; // rax
  int v8; // r10d
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  bool v17; // al
  __int64 v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+20h] [rbp-50h]
  __int64 v20; // [rsp+28h] [rbp-48h]
  int v21; // [rsp+30h] [rbp-40h]
  bool v22; // [rsp+34h] [rbp-3Ch]

  v4 = (_QWORD *)(a3 + 48);
  v5 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v5 == v4 )
    goto LABEL_26;
  if ( !v5 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
LABEL_26:
    BUG();
  v6 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)(v5 - 24) != 31 )
  {
    result = 0;
    if ( *(_BYTE *)v6 != 84 )
      return result;
    return 0;
  }
  if ( *(_BYTE *)v6 != 84 )
    return 0;
  if ( (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  if ( a3 != *(_QWORD *)(v6 + 40) )
    return 0;
  v8 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
  if ( !v8 )
    return 0;
  v9 = *(_QWORD *)(a2 - 32);
  v10 = 0;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v6 - 8);
    v12 = *(_QWORD **)(v11 + 32 * v10);
    if ( *(_BYTE *)v12 == 86 )
    {
      v13 = *(_QWORD *)(v11 + 32LL * *(unsigned int *)(v6 + 72) + 8 * v10);
      if ( v13 == v12[5] )
      {
        v14 = v12[2];
        if ( v14 )
        {
          if ( !*(_QWORD *)(v14 + 8) )
          {
            v15 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v15 == v13 + 48 )
              goto LABEL_28;
            if ( !v15 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
LABEL_28:
              BUG();
            if ( *(_BYTE *)(v15 - 24) == 31 && (*(_DWORD *)(v15 - 20) & 0x7FFFFFF) == 1 )
            {
              v21 = v8;
              v18 = v13;
              v19 = v9;
              v20 = sub_22CF6C0(*(__int64 **)(a1 + 32), *(_WORD *)(a2 + 2) & 0x3F, *(v12 - 8), v9, v13, a3, a2);
              v16 = sub_22CF6C0(*(__int64 **)(a1 + 32), *(_WORD *)(a2 + 2) & 0x3F, *(v12 - 4), v19, v18, a3, a2);
              v8 = v21;
              v9 = v19;
              v17 = (v16 | v20) != 0 && v20 != v16;
              if ( v17 )
                break;
            }
          }
        }
      }
    }
    if ( v8 == (_DWORD)++v10 )
      return 0;
  }
  v22 = v17;
  sub_27DD6D0((__int64 *)a1, v18, a3, v12, v6, v10);
  return v22;
}
