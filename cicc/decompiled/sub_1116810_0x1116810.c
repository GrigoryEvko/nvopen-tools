// Function: sub_1116810
// Address: 0x1116810
//
__int64 __fastcall sub_1116810(
        __int64 a1,
        __int64 a2,
        unsigned __int8 **a3,
        unsigned __int8 **a4,
        __int64 *a5,
        _QWORD *a6,
        __int64 *a7)
{
  _BYTE *v7; // rdi
  __int64 result; // rax
  unsigned __int8 *v9; // rax
  unsigned __int8 *v11; // rax
  int v15; // eax
  _BYTE *v16; // rdx
  __int64 v17; // r8
  _BYTE **v18; // rdx
  _BYTE *v19; // rdi
  unsigned __int8 *v20; // r14
  unsigned int v21; // esi
  __int64 v22; // rdi
  _BYTE *v23; // rax
  _BYTE *v24; // rax
  int v25; // edx
  __int64 v26; // rax
  unsigned __int8 *v27; // [rsp-70h] [rbp-70h]
  __int64 v28; // [rsp-68h] [rbp-68h]
  _QWORD v29[2]; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int8 v30; // [rsp-48h] [rbp-48h]

  v7 = *(_BYTE **)(a2 - 96);
  if ( *v7 != 82 )
    return 0;
  v9 = (unsigned __int8 *)*((_QWORD *)v7 - 8);
  if ( !v9 )
    return 0;
  *a3 = v9;
  v11 = (unsigned __int8 *)*((_QWORD *)v7 - 4);
  if ( !v11 )
    return 0;
  *a4 = v11;
  v15 = sub_B53900((__int64)v7);
  if ( (unsigned int)(v15 - 32) > 1 )
    return 0;
  v16 = *(_BYTE **)(a2 - 64);
  v17 = *(_QWORD *)(a2 - 32);
  if ( v15 == 33 )
  {
    v16 = *(_BYTE **)(a2 - 32);
    v17 = *(_QWORD *)(a2 - 64);
  }
  if ( *v16 != 17 )
    return 0;
  *a6 = v16;
  if ( *(_BYTE *)v17 != 86 )
    return 0;
  v18 = (*(_BYTE *)(v17 + 7) & 0x40) != 0
      ? *(_BYTE ***)(v17 - 8)
      : (_BYTE **)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
  v19 = *v18;
  if ( **v18 != 82 )
    return 0;
  v20 = (unsigned __int8 *)*((_QWORD *)v19 - 8);
  v28 = v17;
  if ( !v20 )
    return 0;
  v27 = (unsigned __int8 *)*((_QWORD *)v19 - 4);
  if ( !v27 )
    return 0;
  v21 = sub_B53900((__int64)v19);
  v22 = (*(_BYTE *)(v28 + 7) & 0x40) != 0 ? *(_QWORD *)(v28 - 8) : v28 - 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF);
  v23 = *(_BYTE **)(v22 + 32);
  if ( *v23 != 17 )
    return 0;
  *a5 = (__int64)v23;
  v24 = *(_BYTE **)(sub_986520(v28) + 64);
  if ( *v24 != 17 )
    return 0;
  *a7 = (__int64)v24;
  if ( *a3 == v20 )
  {
    v25 = v21;
  }
  else
  {
    v25 = sub_B52F50(v21);
    if ( *a3 != v27 )
      return 0;
    v27 = v20;
  }
  if ( v25 != 38 )
  {
    if ( v25 == 40 )
      goto LABEL_26;
    return 0;
  }
  if ( *v27 > 0x15u )
    return 0;
  sub_98FF80((__int64)v29, 0x26u, v27);
  result = v30;
  if ( !v30 )
    return result;
  v27 = (unsigned __int8 *)v29[1];
  v26 = *a5;
  *a5 = *a7;
  *a7 = v26;
LABEL_26:
  result = 1;
  if ( *a4 != v27 )
    return 0;
  return result;
}
