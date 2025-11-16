// Function: sub_28C2900
// Address: 0x28c2900
//
unsigned __int8 *__fastcall sub_28C2900(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // rbx
  __int64 v8; // r12
  unsigned __int8 *result; // rax
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int64 v15; // r12
  __int16 v16; // ax
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 *v22; // rbx
  __int64 *v23; // r12
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 v26; // [rsp+8h] [rbp-78h] BYREF
  __int64 v27; // [rsp+10h] [rbp-70h] BYREF
  __int64 v28; // [rsp+18h] [rbp-68h] BYREF
  __int64 *v29; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v30; // [rsp+28h] [rbp-58h]
  _QWORD v31[10]; // [rsp+30h] [rbp-50h] BYREF

  v26 = a2;
  v29 = &v27;
  v27 = 0;
  v28 = 0;
  v30 = &v28;
  if ( (unsigned __int8)sub_BD3660(a3, 3) )
    return 0;
  for ( i = *(_QWORD *)(a3 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v8 = *(_QWORD *)(i + 24);
    if ( v8 != v26
      && (!(unsigned __int8)sub_BD36B0(*(_QWORD *)(i + 24)) || *(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL) != v26) )
    {
      return 0;
    }
  }
  v10 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x1Cu )
    return 0;
  if ( v10 == 85 )
  {
    v18 = *(_QWORD *)(a3 - 32);
    if ( v18 )
    {
      if ( !*(_BYTE *)v18
        && *(_QWORD *)(v18 + 24) == *(_QWORD *)(a3 + 80)
        && (*(_BYTE *)(v18 + 33) & 0x20) != 0
        && *(_DWORD *)(v18 + 36) == 329 )
      {
        v19 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
        v20 = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
        if ( v19 )
        {
          if ( v20 )
          {
            *v29 = v19;
            *v30 = v20;
            goto LABEL_29;
          }
          *v29 = v19;
        }
      }
    }
    return 0;
  }
  if ( v10 != 86 )
    return 0;
  v11 = *(_QWORD *)(a3 - 96);
  if ( *(_BYTE *)v11 != 82 )
    return 0;
  v12 = *(_QWORD *)(a3 - 64);
  v13 = *(_QWORD *)(v11 - 64);
  v14 = *(_QWORD *)(a3 - 32);
  v15 = *(_QWORD *)(v11 - 32);
  if ( v12 == v13 && v14 == v15 )
  {
    v16 = *(_WORD *)(v11 + 2);
  }
  else
  {
    if ( v12 != v15 || v14 != v13 )
      return 0;
    v16 = *(_WORD *)(v11 + 2);
    if ( v12 != v13 )
    {
      v17 = sub_B52870(v16 & 0x3F);
      goto LABEL_17;
    }
  }
  v17 = v16 & 0x3F;
LABEL_17:
  if ( (unsigned int)(v17 - 38) > 1 )
    return 0;
  if ( !v13 )
    return 0;
  *v29 = v13;
  if ( !v15 )
    return 0;
  *v30 = v15;
LABEL_29:
  v21 = *(_QWORD *)(a1 + 24);
  v31[1] = a1;
  v31[0] = &v29;
  v31[2] = &v26;
  v22 = sub_DD8400(v21, v27);
  v23 = sub_DD8400(*(_QWORD *)(a1 + 24), v28);
  v24 = sub_DD8400(*(_QWORD *)(a1 + 24), a4);
  v25 = (__int64)v24;
  if ( v23 == v24 || (result = sub_28C2740((__int64)v31, (__int64)v22, (__int64)v24, v28)) == 0 )
  {
    if ( v22 != (__int64 *)v25 )
      return sub_28C2740((__int64)v31, v25, (__int64)v23, v27);
    return 0;
  }
  return result;
}
