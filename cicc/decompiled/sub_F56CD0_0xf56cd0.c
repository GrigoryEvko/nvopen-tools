// Function: sub_F56CD0
// Address: 0xf56cd0
//
unsigned __int8 *__fastcall sub_F56CD0(const char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // r14
  char v8; // al
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // r12
  const char *v13; // rsi
  __int64 *v14; // r8
  __int64 v15; // rsi
  unsigned __int8 *v16; // rsi
  __int64 v17; // rdx
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 *v24; // r15
  __int64 v25; // rsi
  __int64 *v26; // [rsp+8h] [rbp-88h]
  int v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  __int64 *v30; // [rsp+20h] [rbp-70h]
  const char *v31; // [rsp+30h] [rbp-60h] BYREF
  __int64 v32; // [rsp+38h] [rbp-58h]
  __int16 v33; // [rsp+50h] [rbp-40h]

  v6 = *((_QWORD *)a1 + 6) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (const char *)v6 == a1 + 48 )
    goto LABEL_41;
  if ( !v6 )
    goto LABEL_41;
  v7 = v6 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
    goto LABEL_41;
  v8 = *(_BYTE *)(v6 - 24);
  if ( v8 == 34 )
    return sub_F565E0(v6 - 24, a2, a3, a4, a5, a6);
  if ( v8 != 37 )
  {
    if ( v8 == 39 )
    {
      v33 = 261;
      v31 = sub_BD5D20(v6 - 24);
      v32 = v17;
      v18 = *(_DWORD *)(v6 - 20);
      if ( (*(_BYTE *)(v6 - 22) & 1) != 0 )
        v19 = (v18 & 0x7FFFFFF) - 2;
      else
        v19 = (v18 & 0x7FFFFFF) - 1;
      v28 = v19;
      v20 = **(_QWORD **)(v6 - 32);
      v21 = sub_BD2DA0(80);
      v12 = v21;
      if ( v21 )
        sub_B4C2D0(v21, v20, 0, v28, (__int64)&v31, 0, v6, 0);
      v22 = *(_QWORD *)(v6 - 32);
      v30 = (__int64 *)(v22 + 32LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF));
      if ( (*(_BYTE *)(v6 - 22) & 1) != 0 )
      {
        v23 = (__int64 *)(v22 + 64);
        if ( v22 + 32LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) == v22 + 64 )
        {
LABEL_35:
          v29 = *(_QWORD *)(v22 + 32);
          goto LABEL_11;
        }
      }
      else
      {
        v23 = (__int64 *)(v22 + 32);
        if ( v30 == (__int64 *)(v22 + 32) )
          goto LABEL_39;
      }
      v24 = v23;
      do
      {
        v25 = *v24;
        v24 += 4;
        sub_B4C4B0(v12, v25);
      }
      while ( v30 != v24 );
      if ( (*(_BYTE *)(v6 - 22) & 1) != 0 )
      {
        v22 = *(_QWORD *)(v6 - 32);
        goto LABEL_35;
      }
LABEL_39:
      v29 = 0;
      goto LABEL_11;
    }
LABEL_41:
    BUG();
  }
  v10 = *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF));
  v11 = sub_BD2C40(72, 1u);
  v12 = (__int64)v11;
  if ( v11 )
    sub_B4BF70((__int64)v11, v10, 0, 1u, v6, 0);
  if ( (*(_BYTE *)(v6 - 22) & 1) != 0 )
    v29 = *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v6 - 20) & 0x7FFFFFF)));
  else
    v29 = 0;
LABEL_11:
  sub_BD6B90((unsigned __int8 *)v12, (unsigned __int8 *)(v6 - 24));
  v13 = *(const char **)(v6 + 24);
  v14 = (__int64 *)(v12 + 48);
  v31 = v13;
  if ( v13 )
  {
    sub_B96E90((__int64)&v31, (__int64)v13, 1);
    v14 = (__int64 *)(v12 + 48);
    if ( (const char **)(v12 + 48) == &v31 )
    {
      if ( v31 )
        sub_B91220((__int64)&v31, (__int64)v31);
      goto LABEL_15;
    }
    v15 = *(_QWORD *)(v12 + 48);
    if ( !v15 )
    {
LABEL_22:
      v16 = (unsigned __int8 *)v31;
      *(_QWORD *)(v12 + 48) = v31;
      if ( v16 )
        sub_B976B0((__int64)&v31, v16, (__int64)v14);
      goto LABEL_15;
    }
LABEL_21:
    v26 = v14;
    sub_B91220((__int64)v14, v15);
    v14 = v26;
    goto LABEL_22;
  }
  if ( v14 != (__int64 *)&v31 )
  {
    v15 = *(_QWORD *)(v12 + 48);
    if ( v15 )
      goto LABEL_21;
  }
LABEL_15:
  sub_AA5980(v29, (__int64)a1, 0);
  sub_BD84D0(v6 - 24, v12);
  sub_B43D60((_QWORD *)(v6 - 24));
  if ( a2 )
  {
    v31 = a1;
    v32 = v29 | 4;
    sub_FFB3D0(a2, &v31, 1);
  }
  return (unsigned __int8 *)v12;
}
