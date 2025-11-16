// Function: sub_AA8550
// Address: 0xaa8550
//
__int64 __fastcall sub_AA8550(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v10; // r8
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // r13
  __int64 *v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+28h] [rbp-48h] BYREF
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int16 v29; // [rsp+38h] [rbp-38h]

  if ( a5 )
    return sub_AA8210((__int64)a1, a2, a3, a4);
  v10 = a1[4];
  if ( v10 == a1[9] + 72LL || !v10 )
    v11 = 0;
  else
    v11 = v10 - 24;
  v23 = v11;
  v24 = a1[9];
  v25 = sub_AA48A0((__int64)a1);
  v12 = sub_22077B0(80);
  v13 = v12;
  if ( v12 )
    sub_AA4D50(v12, v25, a4, v24, v23);
  v14 = a2 - 3;
  if ( !a2 )
    v14 = 0;
  v15 = *(_QWORD *)sub_B46C60(v14);
  v27 = v15;
  if ( v15 )
    sub_B96E90(&v27, v15, 1);
  sub_AA80F0(v13, (unsigned __int64 *)(v13 + 48), 0, (__int64)a1, a2, a3, a1 + 6, 0);
  sub_B43C20(&v28, a1);
  v16 = v29;
  v26 = v28;
  v17 = sub_BD2C40(72, 1);
  if ( v17 )
    sub_B4C8F0(v17, v13, 1, v26, v16);
  v18 = (__int64 *)(v17 + 48);
  v28 = v27;
  if ( v27 )
  {
    sub_B96E90(&v28, v27, 1);
    if ( v18 == &v28 )
    {
      if ( v28 )
        sub_B91220(&v28);
      goto LABEL_18;
    }
    if ( !*(_QWORD *)(v17 + 48) )
    {
LABEL_24:
      v22 = v28;
      *(_QWORD *)(v17 + 48) = v28;
      if ( v22 )
        sub_B976B0(&v28, v22, v17 + 48, v19, v20, v21);
      goto LABEL_18;
    }
LABEL_23:
    sub_B91220(v17 + 48);
    goto LABEL_24;
  }
  if ( v18 != &v28 && *(_QWORD *)(v17 + 48) )
    goto LABEL_23;
LABEL_18:
  sub_AA5DE0(v13, (__int64)a1, v13);
  if ( v27 )
    sub_B91220(&v27);
  return v13;
}
