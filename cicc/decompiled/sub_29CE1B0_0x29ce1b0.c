// Function: sub_29CE1B0
// Address: 0x29ce1b0
//
__int64 __fastcall sub_29CE1B0(__int64 a1, char a2)
{
  const char *v2; // r15
  const char *v4; // rbx
  size_t v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r8d
  unsigned __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rbx
  unsigned __int8 *v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 *v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  int v26; // esi
  __int64 v27; // rax
  __int64 *v28; // rdi
  _QWORD *v29; // rax
  unsigned __int8 *v30; // rsi
  __int64 v31; // rdi
  __int16 v32; // dx
  __int64 v33; // rcx
  unsigned __int8 v34; // r8
  char v35; // r9
  __int64 v36; // [rsp+8h] [rbp-78h]
  const char *v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  size_t v39; // [rsp+18h] [rbp-68h]
  unsigned __int64 v40; // [rsp+20h] [rbp-60h]
  unsigned __int64 v41; // [rsp+20h] [rbp-60h]
  unsigned __int64 v42; // [rsp+28h] [rbp-58h]
  unsigned __int8 v43; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v44; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int8 *v45; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int8 *v46[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = "instrument-function-exit-inlined";
  v4 = "instrument-function-entry-inlined";
  if ( !a2 )
    v4 = "instrument-function-entry";
  v5 = (-(__int64)(a2 == 0) & 0xFFFFFFFFFFFFFFF8LL) + 33;
  if ( !a2 )
    v2 = "instrument-function-exit";
  v39 = strlen(v2);
  v46[0] = (unsigned __int8 *)sub_B2D7E0(a1, v4, v5);
  v6 = sub_A72240((__int64 *)v46);
  v40 = v7;
  v36 = v6;
  v46[0] = (unsigned __int8 *)sub_B2D7E0(a1, v2, v39);
  v8 = sub_A72240((__int64 *)v46);
  v9 = 0;
  v38 = v8;
  v42 = v10;
  if ( !v40 )
    goto LABEL_6;
  v45 = 0;
  v24 = sub_B92180(a1);
  v25 = v24;
  if ( v24 )
  {
    v26 = *(_DWORD *)(v24 + 20);
    v27 = *(_QWORD *)(v24 + 8);
    v28 = (__int64 *)(v27 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v27 & 4) != 0 )
      v28 = (__int64 *)*v28;
    v29 = sub_B01860(v28, v26, 0, v25, 0, 0, 0, 1);
    sub_B10CB0(v46, (__int64)v29);
    v45 = v46[0];
    if ( !v46[0] )
      goto LABEL_50;
    sub_B976B0((__int64)v46, v46[0], (__int64)&v45);
    v30 = v45;
  }
  else
  {
    v30 = v45;
  }
  v46[0] = v30;
  if ( v30 )
    sub_B96E90((__int64)v46, (__int64)v30, 1);
LABEL_50:
  v31 = *(_QWORD *)(a1 + 80);
  if ( v31 )
    v31 -= 24;
  v33 = sub_AA5190(v31);
  if ( v33 )
  {
    v34 = v32;
    v35 = HIBYTE(v32);
  }
  else
  {
    v35 = 0;
    v34 = 0;
  }
  sub_29CD440(a1, v36, v40, v33, v34, v35, (__int64 *)v46);
  if ( v46[0] )
    sub_B91220((__int64)v46, (__int64)v46[0]);
  sub_B2D4A0(a1, v4, v5);
  if ( v45 )
    sub_B91220((__int64)&v45, (__int64)v45);
  v9 = 1;
LABEL_6:
  if ( !v42 )
    return v9;
  if ( *(_QWORD *)(a1 + 80) == a1 + 72 )
    goto LABEL_28;
  v37 = v2;
  v11 = *(_QWORD *)(a1 + 80);
  v12 = a1 + 72;
  do
  {
    while ( 1 )
    {
      if ( !v11 )
        BUG();
      v13 = *(_QWORD *)(v11 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == v11 + 24 )
        goto LABEL_63;
      if ( !v13 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_63:
        BUG();
      if ( *(_BYTE *)(v13 - 24) == 30 )
        break;
      v11 = *(_QWORD *)(v11 + 8);
      if ( v12 == v11 )
        goto LABEL_27;
    }
    v41 = v13 - 24;
    v14 = sub_AA4E50(v11 - 24);
    v44 = 0;
    v15 = v14;
    if ( !v14 )
      v15 = v41;
    v16 = *(unsigned __int8 **)(v15 + 48);
    v45 = v16;
    if ( v16 && (sub_B96E90((__int64)&v45, (__int64)v16, 1), (v17 = v45) != 0) )
    {
      if ( v44 )
      {
        sub_B91220((__int64)&v44, (__int64)v44);
        v17 = v45;
        v44 = v45;
        if ( !v45 )
        {
          v46[0] = 0;
          goto LABEL_22;
        }
      }
      else
      {
        v44 = v45;
      }
      sub_B96E90((__int64)&v44, (__int64)v17, 1);
    }
    else
    {
      v19 = sub_B92180(a1);
      v20 = v19;
      if ( v19 )
      {
        v21 = *(_QWORD *)(v19 + 8);
        v22 = (__int64 *)(v21 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v21 & 4) != 0 )
          v22 = (__int64 *)*v22;
        v23 = sub_B01860(v22, 0, 0, v20, 0, 0, 0, 1);
        sub_B10CB0(v46, (__int64)v23);
        if ( v44 )
          sub_B91220((__int64)&v44, (__int64)v44);
        v44 = v46[0];
        if ( v46[0] )
          sub_B976B0((__int64)v46, v46[0], (__int64)&v44);
      }
    }
    if ( v45 )
      sub_B91220((__int64)&v45, (__int64)v45);
    v46[0] = v44;
    if ( v44 )
      sub_B96E90((__int64)v46, (__int64)v44, 1);
LABEL_22:
    sub_29CD440(a1, v38, v42, v15 + 24, 0, 0, (__int64 *)v46);
    if ( v46[0] )
      sub_B91220((__int64)v46, (__int64)v46[0]);
    if ( v44 )
      sub_B91220((__int64)&v44, (__int64)v44);
    v11 = *(_QWORD *)(v11 + 8);
    LOBYTE(v9) = 1;
  }
  while ( v12 != v11 );
LABEL_27:
  v2 = v37;
LABEL_28:
  v43 = v9;
  sub_B2D4A0(a1, v2, v39);
  return v43;
}
