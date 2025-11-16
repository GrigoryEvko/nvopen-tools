// Function: sub_1180440
// Address: 0x1180440
//
_QWORD *__fastcall sub_1180440(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rcx
  int v8; // eax
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // [rsp+0h] [rbp-90h] BYREF
  __int64 v16; // [rsp+8h] [rbp-88h] BYREF
  _QWORD *v17[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v19[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v20; // [rsp+50h] [rbp-40h]

  v1 = *(_QWORD *)(a1 - 96);
  v2 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v1 != 93 )
    return 0;
  if ( *(_DWORD *)(v1 + 80) != 1 )
    return 0;
  if ( **(_DWORD **)(v1 + 72) != 1 )
    return 0;
  v5 = *(_QWORD *)(v1 - 32);
  if ( *(_BYTE *)v5 != 85 )
    return 0;
  v6 = *(_QWORD *)(v5 - 32);
  if ( !v6 )
    return 0;
  if ( *(_BYTE *)v6 )
    return 0;
  v7 = *(_QWORD *)(v5 + 80);
  if ( *(_QWORD *)(v6 + 24) != v7 || (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
    return 0;
  v8 = *(_DWORD *)(v6 + 36);
  if ( v8 != 312 )
  {
    switch ( v8 )
    {
      case 333:
      case 339:
      case 360:
      case 369:
      case 372:
        break;
      default:
        return 0;
    }
  }
  if ( *(_BYTE *)v2 != 93 || *(_DWORD *)(v2 + 80) != 1 || **(_DWORD **)(v2 + 72) || v5 != *(_QWORD *)(v2 - 32) )
    return 0;
  v9 = *(_QWORD *)(a1 - 64);
  v15 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  v16 = *(_QWORD *)(v5 + 32 * (1LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)));
  v17[0] = &v15;
  v17[1] = &v16;
  v10 = *(_QWORD *)(v5 - 32);
  if ( !v10 || *(_BYTE *)v10 || v7 != *(_QWORD *)(v10 + 24) )
LABEL_49:
    BUG();
  if ( *(_DWORD *)(v10 + 36) == 360 )
  {
    v19[0] = 0;
    if ( (unsigned __int8)sub_995B10(v19, v9) )
    {
      v11 = 359;
      goto LABEL_32;
    }
    v10 = *(_QWORD *)(v5 - 32);
  }
  if ( !v10 )
    goto LABEL_36;
  if ( *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(v5 + 80) )
    goto LABEL_49;
  if ( *(_DWORD *)(v10 + 36) == 372 )
  {
    if ( (unsigned __int8)sub_1178DE0(v9) )
    {
      v11 = 371;
      goto LABEL_32;
    }
    v10 = *(_QWORD *)(v5 - 32);
    if ( v10 )
    {
      if ( *(_BYTE *)v10 )
        goto LABEL_49;
      goto LABEL_25;
    }
LABEL_36:
    BUG();
  }
LABEL_25:
  if ( *(_QWORD *)(v10 + 24) != *(_QWORD *)(v5 + 80) )
    goto LABEL_49;
  if ( *(_DWORD *)(v10 + 36) == 312 )
  {
    if ( (unsigned __int8)sub_117FD30(v17, v9, 1) )
    {
      v11 = 311;
      goto LABEL_32;
    }
    v10 = *(_QWORD *)(v5 - 32);
    if ( !v10 )
      goto LABEL_49;
  }
  if ( *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(v5 + 80) )
    goto LABEL_49;
  if ( *(_DWORD *)(v10 + 36) != 339 || !(unsigned __int8)sub_117FD30(v17, v9, 0) )
    return 0;
  v11 = 338;
LABEL_32:
  v12 = 0;
  v19[0] = *(_QWORD **)(a1 + 8);
  v13 = (__int64 *)sub_B43CA0(a1);
  v14 = sub_B6E160(v13, v11, (__int64)v19, 1);
  v20 = 257;
  v18[0] = v15;
  v18[1] = v16;
  if ( v14 )
    v12 = *(_QWORD *)(v14 + 24);
  v3 = sub_BD2CC0(88, 3u);
  if ( v3 )
  {
    sub_B44260((__int64)v3, **(_QWORD **)(v12 + 16), 56, 3u, 0, 0);
    v3[9] = 0;
    sub_B4A290((__int64)v3, v12, v14, v18, 2, (__int64)v19, 0, 0);
  }
  return v3;
}
