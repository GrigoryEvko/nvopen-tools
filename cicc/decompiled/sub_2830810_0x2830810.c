// Function: sub_2830810
// Address: 0x2830810
//
__int64 __fastcall sub_2830810(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // r14
  __int64 v3; // rcx
  __int64 *v4; // r8
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  _QWORD *v11; // rdx
  unsigned __int64 v12; // rax
  _BYTE *v13; // rdx
  unsigned __int8 *v14; // rbx
  unsigned __int8 *v15; // r13
  unsigned __int8 **v16; // rsi
  __int64 v17; // rdx
  unsigned int v18; // r12d
  __int64 *v19; // rax
  unsigned int v20; // eax
  char v21; // al
  __int64 *v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v25; // [rsp+18h] [rbp-68h] BYREF
  unsigned __int8 *v26; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v27; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v28[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 (__fastcall *v29)(const __m128i **, const __m128i *, int); // [rsp+40h] [rbp-40h]
  __int64 (__fastcall *v30)(_QWORD *, unsigned __int8 **); // [rsp+48h] [rbp-38h]

  v1 = sub_D4B130(*(_QWORD *)(a1 + 8));
  v2 = *(__int64 **)(a1 + 96);
  v3 = v1;
  v4 = &v2[*(unsigned int *)(a1 + 104)];
  if ( v2 != v4 )
  {
    while ( 1 )
    {
      v5 = *v2;
      if ( (*(_DWORD *)(*v2 + 4) & 0x7FFFFFF) != 0 )
        break;
LABEL_8:
      if ( v4 == ++v2 )
        goto LABEL_9;
    }
    v6 = 0;
    v7 = 8LL * (*(_DWORD *)(*v2 + 4) & 0x7FFFFFF);
    while ( 1 )
    {
      v8 = *(_QWORD *)(v5 - 8);
      v9 = *(_BYTE **)(v8 + 4 * v6);
      if ( *v9 > 0x15u )
      {
        if ( *v9 <= 0x1Cu )
          return 0;
        v10 = 32LL * *(unsigned int *)(v5 + 72);
        if ( v3 == *(_QWORD *)(v10 + v8 + v6) )
        {
          v23 = v4;
          v24 = v3;
          v21 = sub_D48480(*(_QWORD *)a1, (__int64)v9, v10, v3);
          v3 = v24;
          v4 = v23;
          if ( !v21 )
            return 0;
        }
      }
      v6 += 8;
      if ( v6 == v7 )
        goto LABEL_8;
    }
  }
LABEL_9:
  v11 = (_QWORD *)(sub_D47930(*(_QWORD *)(a1 + 8)) + 48);
  v12 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v12 == v11 || !v12 || (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 > 0xA )
    BUG();
  if ( *(_BYTE *)(v12 - 24) != 31 )
    BUG();
  if ( (*(_DWORD *)(v12 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v13 = *(_BYTE **)(v12 - 120);
  if ( (unsigned __int8)(*v13 - 82) > 1u )
    return 1;
  v14 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
  v15 = (unsigned __int8 *)*((_QWORD *)v13 - 4);
  v29 = sub_282F860;
  v30 = sub_2830610;
  v16 = &v27;
  v28[0] = a1;
  v28[1] = v28;
  v27 = v14;
  if ( (unsigned __int8)sub_2830610(v28, &v27) )
  {
    v25 = v15;
    if ( !v29 )
      goto LABEL_33;
    v16 = &v25;
    v18 = v30(v28, &v25);
    if ( (_BYTE)v18 )
      goto LABEL_23;
  }
  v26 = v14;
  if ( !v29 )
    goto LABEL_33;
  v16 = &v26;
  if ( (unsigned __int8)v30(v28, &v26) && *v14 > 0x15u )
  {
LABEL_21:
    v19 = sub_DD8400(*(_QWORD *)(a1 + 16), (__int64)v15);
    LOBYTE(v20) = sub_DADE90(*(_QWORD *)(a1 + 16), (__int64)v19, *(_QWORD *)a1);
    v18 = v20;
    if ( (_BYTE)v20 )
      goto LABEL_23;
    goto LABEL_22;
  }
  v27 = v15;
  if ( !v29 )
LABEL_33:
    sub_4263D6(v28, v16, v17);
  if ( (unsigned __int8)v30(v28, &v27) && *v15 > 0x15u )
  {
    v15 = v14;
    goto LABEL_21;
  }
LABEL_22:
  v18 = 0;
LABEL_23:
  if ( v29 )
    v29((const __m128i **)v28, (const __m128i *)v28, 3);
  return v18;
}
