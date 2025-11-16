// Function: sub_18F3020
// Address: 0x18f3020
//
__int64 __fastcall sub_18F3020(__int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned int v10; // eax
  unsigned int v11; // r8d
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r15
  _QWORD *v24; // rax
  _QWORD *v25; // rbx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // r10
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rcx
  unsigned __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rcx
  int v37; // [rsp+4h] [rbp-7Ch]
  __int64 v38; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+10h] [rbp-70h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  __int64 *v42; // [rsp+28h] [rbp-58h] BYREF
  char v43[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v44; // [rsp+40h] [rbp-40h]

  v8 = a4;
  v9 = a4 + a5;
  v10 = sub_15603A0((_QWORD *)(a1 + 56), 0);
  if ( !(_BYTE)a6 )
    v8 = v9;
  if ( !v8 )
  {
    v11 = 0;
    if ( !v10 )
      return v11;
    goto LABEL_9;
  }
  if ( (v8 & (v8 - 1)) != 0 || v10 > v8 )
  {
    v11 = 0;
    if ( !v10 || v8 % v10 )
      return v11;
LABEL_9:
    v13 = v8 - *a2;
    if ( (_BYTE)a6 )
      goto LABEL_10;
    goto LABEL_35;
  }
  v13 = v8 - *a2;
  if ( !(_BYTE)a6 )
LABEL_35:
    v13 = *a3 - v13;
LABEL_10:
  v14 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(_BYTE *)(a1 + 16) != 78 )
    goto LABEL_11;
  v34 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v34 + 16)
    || (*(_BYTE *)(v34 + 33) & 0x20) == 0
    || (unsigned int)(*(_DWORD *)(v34 + 36) - 134) > 4
    || ((1LL << (*(_BYTE *)(v34 + 36) + 122)) & 0x15) == 0 )
  {
    goto LABEL_11;
  }
  v35 = *(_QWORD *)(a1 + 24 * (3 - v14));
  v36 = *(_QWORD **)(v35 + 24);
  if ( *(_DWORD *)(v35 + 32) > 0x40u )
    v36 = (_QWORD *)*v36;
  v11 = 0;
  if ( !(v13 % (unsigned int)v36) )
  {
LABEL_11:
    v39 = v13;
    v15 = *(__int64 **)(a1 + 24 * (2 - v14));
    v16 = sub_15A0680(*v15, v13, 0);
    v17 = (__int64 *)(a1 + 24 * (2LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( *v17 )
    {
      v18 = v17[1];
      v19 = v17[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v19 = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
    }
    *v17 = v16;
    if ( v16 )
    {
      v20 = *(_QWORD *)(v16 + 8);
      v17[1] = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = (unsigned __int64)(v17 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
      v17[2] = (v16 + 8) | v17[2] & 3;
      *(_QWORD *)(v16 + 8) = v17;
    }
    *a3 = v39;
    v11 = a6;
    if ( !(_BYTE)a6 )
    {
      v40 = v8 - *a2;
      v42 = (__int64 *)sub_15A0680(*v15, v40, 0);
      v44 = 257;
      v21 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v22 = *(_QWORD *)v21;
      if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) == 16 )
        v22 = **(_QWORD **)(v22 + 16);
      v23 = *(_QWORD *)(v22 + 24);
      v24 = sub_1648A60(72, 2u);
      v25 = v24;
      if ( v24 )
      {
        v38 = (__int64)(v24 - 6);
        v26 = *(_QWORD *)v21;
        if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) == 16 )
          v26 = **(_QWORD **)(v26 + 16);
        v37 = *(_DWORD *)(v26 + 8) >> 8;
        v27 = (__int64 *)sub_15F9F50(v23, (__int64)&v42, 1);
        v28 = (__int64 *)sub_1646BA0(v27, v37);
        v29 = *(_QWORD *)v21;
        if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) == 16 || (v29 = *v42, *(_BYTE *)(*v42 + 8) == 16) )
          v28 = sub_16463B0(v28, *(_QWORD *)(v29 + 32));
        sub_15F1EA0((__int64)v25, (__int64)v28, 32, v38, 2, a1);
        v25[7] = v23;
        v25[8] = sub_15F9F50(v23, (__int64)&v42, 1);
        sub_15F9CE0((__int64)v25, v21, (__int64 *)&v42, 1, (__int64)v43);
      }
      sub_15FA2E0((__int64)v25, 1);
      v30 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( *v30 )
      {
        v31 = v30[1];
        v32 = v30[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v32 = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = *(_QWORD *)(v31 + 16) & 3LL | v32;
      }
      *v30 = v25;
      if ( v25 )
      {
        v33 = v25[1];
        v30[1] = v33;
        if ( v33 )
          *(_QWORD *)(v33 + 16) = (unsigned __int64)(v30 + 1) | *(_QWORD *)(v33 + 16) & 3LL;
        v30[2] = (unsigned __int64)(v25 + 1) | v30[2] & 3LL;
        v25[1] = v30;
      }
      v11 = 1;
      *a2 += v40;
    }
  }
  return v11;
}
