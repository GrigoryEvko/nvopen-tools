// Function: sub_DA2960
// Address: 0xda2960
//
_QWORD *__fastcall sub_DA2960(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  bool v4; // zf
  __int64 v6; // rsi
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ebx
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned int v13; // r8d
  __int64 v14; // rdi
  __int64 v15; // rdx
  _QWORD *v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // r13
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // [rsp+Ch] [rbp-94h]
  _QWORD *v23; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-88h]
  __int64 v25; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-68h]
  _QWORD *v29; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-58h]
  _QWORD *v31; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-48h]
  _QWORD *v33; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v34; // [rsp+68h] [rbp-38h]

  v3 = (_QWORD *)a2;
  v4 = *(_WORD *)(a2 + 24) == 0;
  v24 = 1;
  v23 = 0;
  v26 = 1;
  v25 = 0;
  if ( !v4 || *(_WORD *)(a3 + 24) )
    return v3;
  v6 = *(_QWORD *)(a2 + 32);
  if ( *(_DWORD *)(v6 + 32) <= 0x40u )
  {
    v24 = *(_DWORD *)(v6 + 32);
    v8 = *(_QWORD *)(a3 + 32);
    v17 = *(_QWORD **)(v6 + 24);
    v10 = *(_DWORD *)(v8 + 32);
    v9 = v8 + 24;
    v23 = v17;
    if ( v10 > 0x40 )
    {
LABEL_6:
      sub_C43990((__int64)&v25, v9);
      v10 = v26;
      goto LABEL_7;
    }
  }
  else
  {
    sub_C43990((__int64)&v23, v6 + 24);
    v8 = *(_QWORD *)(a3 + 32);
    v9 = v8 + 24;
    if ( v26 > 0x40 )
      goto LABEL_6;
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 > 0x40 )
      goto LABEL_6;
  }
  v18 = *(_QWORD *)(v8 + 24);
  v26 = v10;
  v25 = v18;
LABEL_7:
  v11 = v24;
  v12 = (unsigned __int64)v23;
  v13 = v24;
  if ( v24 > 0x40 )
    v12 = v23[(v24 - 1) >> 6];
  if ( (v12 & (1LL << ((unsigned __int8)v24 - 1))) != 0 )
    goto LABEL_18;
  v14 = v25;
  v15 = 1LL << ((unsigned __int8)v10 - 1);
  if ( v10 > 0x40 )
  {
    if ( (*(_QWORD *)(v25 + 8LL * ((v10 - 1) >> 6)) & v15) != 0 )
    {
LABEL_20:
      if ( v14 )
        j_j___libc_free_0_0(v14);
      v13 = v24;
      goto LABEL_23;
    }
    if ( v10 == (unsigned int)sub_C444A0((__int64)&v25) )
    {
LABEL_19:
      v14 = v25;
      goto LABEL_20;
    }
    goto LABEL_13;
  }
  if ( (v15 & v25) == 0 )
  {
    if ( !v25 )
    {
LABEL_48:
      v13 = v11;
      goto LABEL_23;
    }
LABEL_13:
    sub_C4B490((__int64)&v27, (__int64)&v23, (__int64)&v25);
    if ( v28 <= 0x40 )
    {
      if ( !v27 )
      {
LABEL_17:
        v10 = v26;
LABEL_18:
        if ( v10 > 0x40 )
          goto LABEL_19;
        v11 = v24;
        goto LABEL_48;
      }
    }
    else
    {
      v22 = v28;
      if ( v22 == (unsigned int)sub_C444A0((__int64)&v27) )
      {
LABEL_15:
        if ( v27 )
          j_j___libc_free_0_0(v27);
        goto LABEL_17;
      }
    }
    v19 = *(__int64 **)(a1 + 8);
    v30 = v24;
    if ( v24 > 0x40 )
      sub_C43780((__int64)&v29, (const void **)&v23);
    else
      v29 = v23;
    sub_C45EE0((__int64)&v29, &v25);
    v20 = v30;
    v30 = 0;
    v32 = v20;
    v31 = v29;
    sub_C46B40((__int64)&v31, &v27);
    v21 = v32;
    v32 = 0;
    v34 = v21;
    v33 = v31;
    v3 = sub_DA26C0(v19, (__int64)&v33);
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v28 <= 0x40 )
      goto LABEL_17;
    goto LABEL_15;
  }
LABEL_23:
  if ( v13 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  return v3;
}
