// Function: sub_3205E80
// Address: 0x3205e80
//
void __fastcall sub_3205E80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 *v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // rcx
  unsigned __int8 v11; // al
  bool v12; // dl
  _BYTE *v13; // rcx
  char *v14; // rdx
  char *v15; // r8
  __int64 v16; // r15
  __int64 *v17; // rbx
  __int64 *v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rcx
  unsigned __int8 v22; // al
  bool v23; // dl
  _BYTE *v24; // rcx
  char *v25; // rdx
  char *v26; // r8
  __int64 v27; // r15
  __int64 *i; // [rsp+8h] [rbp-68h]
  __int64 *j; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  unsigned __int64 v34[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v35[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *(__int64 **)(a1 + 936);
  for ( i = &v6[2 * *(unsigned int *)(a1 + 944)]; i != v6; v6 += 2 )
  {
    v9 = *v6;
    v11 = *(_BYTE *)(*v6 - 16);
    v10 = *v6 - 16;
    v12 = (v11 & 2) != 0;
    if ( (v11 & 2) != 0 )
      v7 = *(__int64 **)(v9 - 32);
    else
      v7 = (__int64 *)(v10 - 8LL * ((v11 >> 2) & 0xF));
    v16 = *v7;
    v8 = v7[3];
    if ( v8 )
    {
      v30 = *v6 - 16;
      v32 = *v6;
      sub_3205010(a1, v8);
      v9 = v32;
      v10 = v30;
      v11 = *(_BYTE *)(v32 - 16);
      v12 = (v11 & 2) != 0;
    }
    if ( v12 )
    {
      v13 = *(_BYTE **)(*(_QWORD *)(v9 - 32) + 8LL);
      if ( !v13 )
        goto LABEL_15;
    }
    else
    {
      v13 = *(_BYTE **)(v10 - 8LL * ((v11 >> 2) & 0xF) + 8);
      if ( !v13 )
      {
LABEL_15:
        v15 = 0;
        goto LABEL_9;
      }
    }
    v13 = (_BYTE *)sub_B91420((__int64)v13);
    v15 = v14;
LABEL_9:
    sub_3205680((__int64)v34, a1, v16, v13, v15, a6);
    if ( (_QWORD *)v34[0] != v35 )
      j_j___libc_free_0(v34[0]);
  }
  v17 = *(__int64 **)(a1 + 904);
  for ( j = &v17[2 * *(unsigned int *)(a1 + 912)]; j != v17; v17 += 2 )
  {
    v20 = *v17;
    v22 = *(_BYTE *)(*v17 - 16);
    v21 = *v17 - 16;
    v23 = (v22 & 2) != 0;
    if ( (v22 & 2) != 0 )
      v18 = *(__int64 **)(v20 - 32);
    else
      v18 = (__int64 *)(v21 - 8LL * ((v22 >> 2) & 0xF));
    v27 = *v18;
    v19 = v18[3];
    if ( v19 )
    {
      v31 = *v17 - 16;
      v33 = *v17;
      sub_3205010(a1, v19);
      v20 = v33;
      v21 = v31;
      v22 = *(_BYTE *)(v33 - 16);
      v23 = (v22 & 2) != 0;
    }
    if ( v23 )
    {
      v24 = *(_BYTE **)(*(_QWORD *)(v20 - 32) + 8LL);
      if ( !v24 )
        goto LABEL_30;
    }
    else
    {
      v24 = *(_BYTE **)(v21 - 8LL * ((v22 >> 2) & 0xF) + 8);
      if ( !v24 )
      {
LABEL_30:
        v26 = 0;
        goto LABEL_24;
      }
    }
    v24 = (_BYTE *)sub_B91420((__int64)v24);
    v26 = v25;
LABEL_24:
    sub_3205680((__int64)v34, a1, v27, v24, v26, a6);
    if ( (_QWORD *)v34[0] != v35 )
      j_j___libc_free_0(v34[0]);
  }
}
