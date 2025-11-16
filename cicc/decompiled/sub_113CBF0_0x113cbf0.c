// Function: sub_113CBF0
// Address: 0x113cbf0
//
_QWORD *__fastcall sub_113CBF0(__m128i *a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int8 v4; // al
  __int64 v5; // r12
  __int64 v6; // r15
  char v7; // al
  unsigned int v8; // ebx
  __int16 v9; // dx
  __int64 v10; // rdx
  _QWORD *result; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int16 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h] BYREF
  __int64 v25; // [rsp+18h] [rbp-68h] BYREF
  _DWORD v26[4]; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v27; // [rsp+30h] [rbp-50h]
  int v28; // [rsp+38h] [rbp-48h]
  __int64 *v29; // [rsp+40h] [rbp-40h]

  v3 = *(_QWORD *)(a2 - 32);
  v4 = *(_BYTE *)v3;
  v5 = v3 + 24;
  if ( *(_BYTE *)v3 != 17 )
  {
    v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v3 + 8) + 8LL) - 17;
    if ( (unsigned int)v12 > 1 )
      return 0;
    if ( v4 > 0x15u )
    {
LABEL_20:
      if ( (unsigned int)v12 <= 1 && v4 <= 0x15u )
      {
        v14 = sub_AD7630(v3, 1, v12);
        if ( v14 )
        {
          v10 = (__int64)(v14 + 24);
          if ( *v14 == 17 )
            return sub_1116A30((__int64)a1, a2, v10);
        }
      }
      return 0;
    }
    v13 = sub_AD7630(v3, 0, v12);
    if ( !v13 || *v13 != 17 )
      goto LABEL_9;
    v5 = (__int64)(v13 + 24);
  }
  v6 = *(_QWORD *)(a2 - 64);
  v7 = *(_BYTE *)v6;
  if ( (unsigned __int8)(*(_BYTE *)v6 - 42) <= 0x11u )
  {
    result = (_QWORD *)sub_113CA70(a1, a2, *(unsigned __int8 **)(a2 - 64), v5);
    if ( result )
      return result;
    v6 = *(_QWORD *)(a2 - 64);
    v7 = *(_BYTE *)v6;
  }
  if ( v7 == 86 )
  {
    v15 = *(_BYTE **)(a2 - 32);
    if ( *v15 != 17 )
      goto LABEL_6;
    v16 = *(_QWORD *)(a2 + 16);
    if ( !v16 || *(_QWORD *)(v16 + 8) )
      goto LABEL_6;
    result = sub_112A580((__int64)a1, a2, v6, (__int64)v15);
    if ( result )
      return result;
    v6 = *(_QWORD *)(a2 - 64);
    v7 = *(_BYTE *)v6;
  }
  if ( v7 == 67 )
  {
    result = sub_11329C0((__int64)a1, a2, v6, (const void **)v5);
    if ( result )
      return result;
    v6 = *(_QWORD *)(a2 - 64);
    v7 = *(_BYTE *)v6;
  }
  if ( v7 == 85 )
  {
    v17 = *(_QWORD *)(v6 - 32);
    if ( v17 )
    {
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v6 + 80) && (*(_BYTE *)(v17 + 33) & 0x20) != 0 )
      {
        result = sub_111AB00(a1, a2, v6, (unsigned __int64 *)v5);
        if ( result )
          return result;
        v6 = *(_QWORD *)(a2 - 64);
      }
    }
  }
LABEL_6:
  v8 = *(_DWORD *)(v5 + 8);
  if ( v8 <= 0x40 )
  {
    if ( !*(_QWORD *)v5 )
      goto LABEL_8;
LABEL_9:
    v3 = *(_QWORD *)(a2 - 32);
    v4 = *(_BYTE *)v3;
    if ( *(_BYTE *)v3 == 17 )
    {
      v10 = v3 + 24;
      return sub_1116A30((__int64)a1, a2, v10);
    }
    v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v3 + 8) + 8LL) - 17;
    goto LABEL_20;
  }
  if ( v8 != (unsigned int)sub_C444A0(v5) )
    goto LABEL_9;
LABEL_8:
  v9 = *(_WORD *)(a2 + 2);
  if ( (v9 & 0x3Fu) - 32 > 1 )
    goto LABEL_9;
  v18 = *(_QWORD *)(v6 + 16);
  if ( !v18 || *(_QWORD *)(v18 + 8) )
    goto LABEL_9;
  if ( *(_BYTE *)v6 != 93
    || *(_DWORD *)(v6 + 80) != 1
    || **(_DWORD **)(v6 + 72)
    || (v20 = *(_QWORD *)(v6 - 32), *(_BYTE *)v20 != 85)
    || (v21 = *(_QWORD *)(v20 - 32)) == 0
    || *(_BYTE *)v21
    || *(_QWORD *)(v21 + 24) != *(_QWORD *)(v20 + 80)
    || *(_DWORD *)(v21 + 36) != 339
    || !*(_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF))
    || (v24 = *(_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF)), *(_BYTE *)v20 != 85)
    || (v22 = *(_QWORD *)(v20 + 32 * (1LL - (*(_DWORD *)(v20 + 4) & 0x7FFFFFF)))) == 0 )
  {
    v26[0] = 372;
    v27 = &v24;
    v26[2] = 0;
    v28 = 1;
    v29 = &v25;
    if ( (unsigned __int8)sub_111E400((__int64)v26, v6) )
    {
      v9 = *(_WORD *)(a2 + 2);
      goto LABEL_46;
    }
    goto LABEL_9;
  }
  v25 = v22;
LABEL_46:
  LOWORD(v29) = 257;
  v19 = v9 & 0x3F;
  result = sub_BD2C40(72, unk_3F10FD0);
  if ( result )
  {
    v23 = result;
    sub_1113300((__int64)result, v19, v24, v25, (__int64)v26);
    return v23;
  }
  return result;
}
