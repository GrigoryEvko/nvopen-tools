// Function: sub_35DEE40
// Address: 0x35dee40
//
void __fastcall sub_35DEE40(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v8; // rsi
  __int64 v9; // r8
  unsigned __int64 v10; // r9
  _QWORD *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 **v20; // r11
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v23; // rax
  _QWORD *v24; // r15
  _QWORD *v25; // rdx
  __int64 v26; // rdi
  _QWORD *v27; // rax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // r14
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // rax
  __int64 v36; // [rsp+10h] [rbp-B0h]
  __int64 **v37; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  __int64 v39; // [rsp+18h] [rbp-A8h]
  __int64 **v40; // [rsp+18h] [rbp-A8h]
  __int64 v41; // [rsp+18h] [rbp-A8h]
  __int64 v42; // [rsp+18h] [rbp-A8h]
  char v43[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v44; // [rsp+50h] [rbp-70h]
  __int64 v45[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v46; // [rsp+80h] [rbp-40h]

  v4 = *a1;
  if ( !a3 )
    BUG();
  v36 = a3 - 24;
  *(_QWORD *)(v4 + 48) = *(_QWORD *)(a3 + 16);
  *(_QWORD *)(v4 + 56) = a3;
  *(_WORD *)(v4 + 64) = a4;
  v8 = *(_QWORD *)sub_B46C60(a3 - 24);
  v45[0] = v8;
  if ( v8 && (sub_B96E90((__int64)v45, v8, 1), (v9 = v45[0]) != 0) )
  {
    v10 = *(unsigned int *)(v4 + 8);
    v11 = *(_QWORD **)v4;
    v12 = v10;
    v13 = (_QWORD *)(*(_QWORD *)v4 + 16 * v10);
    if ( *(_QWORD **)v4 != v13 )
    {
      while ( *(_DWORD *)v11 )
      {
        v11 += 2;
        if ( v13 == v11 )
          goto LABEL_41;
      }
      v11[1] = v45[0];
      goto LABEL_9;
    }
LABEL_41:
    v28 = *(unsigned int *)(v4 + 12);
    if ( v10 >= v28 )
    {
      if ( v28 < ++v10 )
      {
        v41 = v45[0];
        sub_C8D5F0(v4, (const void *)(v4 + 16), v10, 0x10u, v45[0], v10);
        v9 = v41;
        v13 = (_QWORD *)(*(_QWORD *)v4 + 16LL * *(unsigned int *)(v4 + 8));
      }
      *v13 = 0;
      v13[1] = v9;
      ++*(_DWORD *)(v4 + 8);
      v9 = v45[0];
    }
    else
    {
      if ( v13 )
      {
        *(_DWORD *)v13 = 0;
        v13[1] = v9;
        v12 = *(unsigned int *)(v4 + 8);
        v9 = v45[0];
      }
      *(_DWORD *)(v4 + 8) = v12 + 1;
    }
  }
  else
  {
    sub_93FB40(v4, 0);
    v9 = v45[0];
  }
  if ( v9 )
LABEL_9:
    sub_B91220((__int64)v45, v9);
  if ( *(_BYTE *)a2 <= 0x1Cu )
    goto LABEL_19;
  v14 = *(_QWORD *)(a2 + 48);
  v15 = *a1;
  v45[0] = v14;
  if ( v14 && (sub_B96E90((__int64)v45, v14, 1), (v9 = v45[0]) != 0) )
  {
    v10 = *(unsigned int *)(v15 + 8);
    v16 = *(_QWORD **)v15;
    v12 = v10;
    v17 = (_QWORD *)(*(_QWORD *)v15 + 16 * v10);
    if ( *(_QWORD **)v15 != v17 )
    {
      while ( *(_DWORD *)v16 )
      {
        v16 += 2;
        if ( v17 == v16 )
          goto LABEL_48;
      }
      v16[1] = v45[0];
      goto LABEL_18;
    }
LABEL_48:
    v29 = *(unsigned int *)(v15 + 12);
    if ( v10 >= v29 )
    {
      if ( v29 < ++v10 )
      {
        v42 = v45[0];
        sub_C8D5F0(v15, (const void *)(v15 + 16), v10, 0x10u, v45[0], v10);
        v9 = v42;
        v17 = (_QWORD *)(*(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8));
      }
      *v17 = 0;
      v17[1] = v9;
      ++*(_DWORD *)(v15 + 8);
      v9 = v45[0];
    }
    else
    {
      if ( v17 )
      {
        *(_DWORD *)v17 = 0;
        v17[1] = v9;
        LODWORD(v12) = *(_DWORD *)(v15 + 8);
        v9 = v45[0];
      }
      v12 = (unsigned int)(v12 + 1);
      *(_DWORD *)(v15 + 8) = v12;
    }
  }
  else
  {
    sub_93FB40(v15, 0);
    v9 = v45[0];
  }
  if ( v9 )
LABEL_18:
    sub_B91220((__int64)v45, v9);
LABEL_19:
  v18 = a1[1];
  v19 = *a1;
  v44 = 257;
  v20 = *(__int64 ***)(v18 + 56);
  if ( v20 == *(__int64 ***)(a2 + 8) )
  {
    v24 = (_QWORD *)a2;
    goto LABEL_26;
  }
  v21 = *(_QWORD *)(v19 + 80);
  v22 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v21 + 120LL);
  if ( v22 != sub_920130 )
  {
    v40 = v20;
    v34 = v22(v21, 39u, (_BYTE *)a2, (__int64)v20);
    v20 = v40;
    v24 = (_QWORD *)v34;
    goto LABEL_25;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v37 = v20;
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v23 = sub_ADAB70(39, a2, v37, 0);
    else
      v23 = sub_AA93C0(0x27u, a2, (__int64)v37);
    v20 = v37;
    v24 = (_QWORD *)v23;
LABEL_25:
    if ( v24 )
      goto LABEL_26;
  }
  v38 = (__int64)v20;
  v46 = 257;
  v30 = sub_BD2C40(72, 1u);
  v24 = v30;
  if ( v30 )
    sub_B515B0((__int64)v30, a2, v38, (__int64)v45, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, char *, _QWORD, _QWORD))(**(_QWORD **)(v19 + 88) + 16LL))(
    *(_QWORD *)(v19 + 88),
    v24,
    v43,
    *(_QWORD *)(v19 + 56),
    *(_QWORD *)(v19 + 64));
  v12 = *(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8);
  v31 = *(_QWORD *)v19;
  v39 = v12;
  while ( v39 != v31 )
  {
    v32 = *(_QWORD *)(v31 + 8);
    v33 = *(_DWORD *)v31;
    v31 += 16;
    sub_B99FD0((__int64)v24, v33, v32);
  }
LABEL_26:
  if ( *(_BYTE *)v24 <= 0x1Cu )
  {
LABEL_40:
    v26 = a1[1];
    goto LABEL_34;
  }
  if ( *(_BYTE *)a2 == 22 )
    sub_B444E0(v24, a3, a4);
  else
    sub_B44530(v24, v36);
  v26 = a1[1];
  if ( !*(_BYTE *)(v26 + 92) )
  {
LABEL_39:
    sub_C8CC70(v26 + 64, (__int64)v24, (__int64)v25, v12, v9, v10);
    goto LABEL_40;
  }
  v27 = *(_QWORD **)(v26 + 72);
  v12 = *(unsigned int *)(v26 + 84);
  v25 = &v27[v12];
  if ( v27 == v25 )
  {
LABEL_38:
    if ( (unsigned int)v12 < *(_DWORD *)(v26 + 80) )
    {
      v12 = (unsigned int)(v12 + 1);
      *(_DWORD *)(v26 + 84) = v12;
      *v25 = v24;
      ++*(_QWORD *)(v26 + 64);
      v26 = a1[1];
      goto LABEL_34;
    }
    goto LABEL_39;
  }
  while ( v24 != (_QWORD *)*v27 )
  {
    if ( v25 == ++v27 )
      goto LABEL_38;
  }
LABEL_34:
  sub_35DE620(v26, a2, (__int64)v24, v12, v9, v10);
}
