// Function: sub_13CCBD0
// Address: 0x13ccbd0
//
__int64 __fastcall sub_13CCBD0(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // r13
  unsigned __int64 *v11; // rcx
  _QWORD *v12; // rsi
  _QWORD *v13; // rax
  char v14; // dl
  unsigned __int8 v15; // al
  __int16 v16; // ax
  __int64 v17; // r12
  __int64 v18; // rax
  _QWORD *v20; // rdi
  _QWORD *v21; // rcx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rdi
  int v25; // eax
  __int64 v26; // rax
  char v27; // [rsp+Fh] [rbp-A1h]
  int v28; // [rsp+14h] [rbp-9Ch] BYREF
  __int64 v29; // [rsp+18h] [rbp-98h] BYREF
  __int64 v30; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-88h]
  __int64 v32; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v33; // [rsp+38h] [rbp-78h]
  _QWORD *v34; // [rsp+40h] [rbp-70h]
  __int64 v35; // [rsp+48h] [rbp-68h]
  int v36; // [rsp+50h] [rbp-60h]
  _QWORD v37[11]; // [rsp+58h] [rbp-58h] BYREF

  v27 = a3;
  v8 = sub_15A9650(a1, *(_QWORD *)*a2, a3, a4, a5, a6);
  v9 = v8;
  if ( *(_BYTE *)(v8 + 8) == 16 )
    v9 = **(_QWORD **)(v8 + 16);
  v31 = *(_DWORD *)(v9 + 8) >> 8;
  if ( v31 > 0x40 )
    sub_16A4EF0(&v30, 0, 0);
  else
    v30 = 0;
  v10 = *a2;
  v36 = 0;
  v33 = v37;
  v34 = v37;
  v35 = 0x100000004LL;
  v37[0] = v10;
  v32 = 1;
  while ( 1 )
  {
    v15 = *(_BYTE *)(v10 + 16);
    if ( v15 <= 0x17u )
    {
      if ( v15 != 5 )
      {
        if ( v15 == 1 )
          __asm { jmp     rax }
        goto LABEL_18;
      }
      v16 = *(_WORD *)(v10 + 18);
      if ( v16 == 32 )
      {
LABEL_26:
        if ( !v27 && (*(_BYTE *)(v10 + 17) & 2) == 0 || !(unsigned __int8)sub_1634900(v10, a1, &v30) )
          goto LABEL_18;
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
        {
LABEL_9:
          v11 = *(unsigned __int64 **)(v10 - 8);
LABEL_10:
          v12 = (_QWORD *)*v11;
          *a2 = *v11;
          goto LABEL_11;
        }
      }
      else
      {
        if ( v16 != 47 )
          goto LABEL_18;
LABEL_8:
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
          goto LABEL_9;
      }
      v11 = (unsigned __int64 *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      goto LABEL_10;
    }
    if ( v15 == 56 )
      goto LABEL_26;
    if ( v15 == 71 )
      goto LABEL_8;
    v22 = v10 | 4;
    if ( v15 != 78 )
    {
      if ( v15 != 29 )
        goto LABEL_18;
      v22 = v10 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v22 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_18;
    v24 = v23 + 56;
    if ( (v22 & 4) == 0 )
    {
      if ( (unsigned __int8)sub_1560490(v24, 38, &v28) )
      {
        v25 = v28;
        if ( v28 )
          goto LABEL_47;
      }
      v26 = *(_QWORD *)(v23 - 72);
      if ( *(_BYTE *)(v26 + 16) )
        goto LABEL_18;
LABEL_50:
      v29 = *(_QWORD *)(v26 + 112);
      if ( !(unsigned __int8)sub_1560490(&v29, 38, &v28) )
        goto LABEL_18;
      v25 = v28;
      if ( !v28 )
        goto LABEL_18;
      goto LABEL_47;
    }
    if ( !(unsigned __int8)sub_1560490(v24, 38, &v28) || (v25 = v28) == 0 )
    {
      v26 = *(_QWORD *)(v23 - 24);
      if ( *(_BYTE *)(v26 + 16) )
        goto LABEL_18;
      goto LABEL_50;
    }
LABEL_47:
    v12 = *(_QWORD **)(v23 + 24 * ((unsigned int)(v25 - 1) - (unsigned __int64)(*(_DWORD *)(v23 + 20) & 0xFFFFFFF)));
    if ( !v12 )
      goto LABEL_18;
    *a2 = (unsigned __int64)v12;
LABEL_11:
    v13 = v33;
    if ( v34 != v33 )
      goto LABEL_12;
    v20 = &v33[HIDWORD(v35)];
    if ( v33 != v20 )
      break;
LABEL_57:
    if ( HIDWORD(v35) < (unsigned int)v35 )
    {
      ++HIDWORD(v35);
      *v20 = v12;
      ++v32;
      goto LABEL_13;
    }
LABEL_12:
    sub_16CCBA0(&v32, v12);
    if ( !v14 )
      goto LABEL_18;
LABEL_13:
    v10 = *a2;
  }
  v21 = 0;
  while ( (_QWORD *)*v13 != v12 )
  {
    if ( *v13 == -2 )
      v21 = v13;
    if ( v20 == ++v13 )
    {
      if ( !v21 )
        goto LABEL_57;
      *v21 = v12;
      --v36;
      ++v32;
      goto LABEL_13;
    }
  }
LABEL_18:
  v17 = sub_15A1070(v9, &v30);
  v18 = *(_QWORD *)*a2;
  if ( *(_BYTE *)(v18 + 8) == 16 )
    v17 = sub_15A0390(*(_QWORD *)(v18 + 32));
  if ( v34 != v33 )
    _libc_free((unsigned __int64)v34);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  return v17;
}
