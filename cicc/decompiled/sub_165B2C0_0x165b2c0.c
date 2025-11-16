// Function: sub_165B2C0
// Address: 0x165b2c0
//
void __fastcall sub_165B2C0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdx
  char v4; // al
  __int64 v5; // r9
  __int64 v6; // r11
  __int64 v7; // rbx
  __int64 v8; // r10
  int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  const char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // ebx
  int v19; // eax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  const char *v27; // rax
  int v28; // [rsp+Ch] [rbp-124h]
  __int64 v29; // [rsp+10h] [rbp-120h]
  __int64 v30; // [rsp+18h] [rbp-118h]
  const char *v31; // [rsp+20h] [rbp-110h] BYREF
  char v32; // [rsp+30h] [rbp-100h]
  char v33; // [rsp+31h] [rbp-FFh]
  __int64 v34[3]; // [rsp+40h] [rbp-F0h] BYREF
  _QWORD *v35; // [rsp+58h] [rbp-D8h]
  _QWORD v36[2]; // [rsp+A0h] [rbp-90h] BYREF
  char v37; // [rsp+B0h] [rbp-80h]
  char v38; // [rsp+B1h] [rbp-7Fh]
  _QWORD *v39; // [rsp+B8h] [rbp-78h]

  v3 = *(_QWORD *)(a2 - 24);
  v4 = *(_BYTE *)(v3 + 16);
  if ( v4 == 20 )
  {
    v38 = 1;
    v15 = "cannot use musttail call with inline asm";
LABEL_13:
    v36[0] = v15;
    v37 = 3;
    sub_164FF40(a1, (__int64)v36);
    if ( *a1 )
      sub_164FA80(a1, a2);
    return;
  }
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 64);
  v7 = *(_QWORD *)(v5 + 56);
  v8 = *(_QWORD *)(v7 + 24);
  if ( v4 || (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
  {
    v9 = *(_DWORD *)(v8 + 12);
    if ( v9 != *(_DWORD *)(v6 + 12) )
    {
      v38 = 1;
      v15 = "cannot guarantee tail call due to mismatched parameter counts";
      goto LABEL_13;
    }
    if ( v9 != 1 )
    {
      v10 = (unsigned int)(v9 - 2);
      v11 = 1;
      v12 = v10 + 2;
      while ( 1 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8 * v11);
        v14 = *(_QWORD *)(*(_QWORD *)(v8 + 16) + 8 * v11);
        if ( v14 != v13
          && (*(_BYTE *)(v14 + 8) != 15
           || *(_BYTE *)(v13 + 8) != 15
           || *(_DWORD *)(v14 + 8) >> 8 != *(_DWORD *)(v13 + 8) >> 8) )
        {
          break;
        }
        if ( v12 == ++v11 )
          goto LABEL_16;
      }
      v38 = 1;
      v15 = "cannot guarantee tail call due to mismatched parameter types";
      goto LABEL_13;
    }
  }
LABEL_16:
  if ( (*(_DWORD *)(v6 + 8) >> 8 != 0) != (*(_DWORD *)(v8 + 8) >> 8 != 0) )
  {
    v38 = 1;
    v15 = "cannot guarantee tail call due to mismatched varargs";
    goto LABEL_13;
  }
  v16 = **(_QWORD **)(v6 + 16);
  v17 = **(_QWORD **)(v8 + 16);
  if ( v17 != v16
    && (*(_BYTE *)(v17 + 8) != 15 || *(_BYTE *)(v16 + 8) != 15 || *(_DWORD *)(v17 + 8) >> 8 != *(_DWORD *)(v16 + 8) >> 8) )
  {
    v38 = 1;
    v15 = "cannot guarantee tail call due to mismatched return types";
    goto LABEL_13;
  }
  if ( ((*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF) != ((*(_WORD *)(v7 + 18) >> 4) & 0x3FF) )
  {
    v38 = 1;
    v15 = "cannot guarantee tail call due to mismatched calling conv";
    goto LABEL_13;
  }
  v30 = *(_QWORD *)(v7 + 112);
  v29 = *(_QWORD *)(a2 + 56);
  v28 = *(_DWORD *)(v8 + 12) - 1;
  if ( *(_DWORD *)(v8 + 12) == 1 )
  {
LABEL_35:
    v21 = *(_QWORD *)(a2 + 32);
    v22 = a2;
    if ( v21 == v5 + 40 || !v21 )
      goto LABEL_51;
    v23 = v21 - 24;
    if ( *(_BYTE *)(v21 - 8) == 71 )
    {
      v24 = *(_QWORD *)(v21 - 48);
      if ( !v24 || a2 != v24 )
      {
        v38 = 1;
        v27 = "bitcast following musttail call must use the call";
LABEL_48:
        v36[0] = v27;
        v37 = 3;
        sub_164FF40(a1, (__int64)v36);
        if ( *a1 )
          sub_164FA80(a1, v23);
        return;
      }
      v25 = *(_QWORD *)(v21 + 8);
      if ( v25 == *(_QWORD *)(v21 + 16) + 40LL || !v25 )
      {
LABEL_51:
        v34[0] = a2;
        v38 = 1;
        v36[0] = "musttail call must precede a ret with an optional bitcast";
        v37 = 3;
        sub_165B1D0(a1, (__int64)v36, v34);
        return;
      }
      v22 = v21 - 24;
      v23 = v25 - 24;
    }
    if ( *(_BYTE *)(v23 + 16) == 25 )
    {
      if ( (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) == 0 )
        return;
      v26 = *(_QWORD *)(v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF));
      if ( !v26 || v26 == v22 )
        return;
      v38 = 1;
      v27 = "musttail call result must be returned";
      goto LABEL_48;
    }
    goto LABEL_51;
  }
  v18 = 0;
  while ( 1 )
  {
    sub_164DB80((__int64)v34, v18, v30);
    sub_164DB80((__int64)v36, v18, v29);
    if ( !(unsigned __int8)sub_1561B70(v34, v36) )
      break;
    ++v18;
    sub_164EC70(v39);
    sub_164EC70(v35);
    if ( v28 == v18 )
    {
      v5 = *(_QWORD *)(a2 + 40);
      goto LABEL_35;
    }
  }
  v19 = *(_DWORD *)(a2 + 20);
  v33 = 1;
  v32 = 3;
  v20 = *(_QWORD *)(a2 + 24 * (v18 - (unsigned __int64)(v19 & 0xFFFFFFF)));
  v31 = "cannot guarantee tail call due to mismatched ABI impacting function attributes";
  sub_164FF40(a1, (__int64)&v31);
  if ( *a1 )
  {
    sub_164FA80(a1, a2);
    sub_164FA80(a1, v20);
  }
  sub_164EC70(v39);
  sub_164EC70(v35);
}
