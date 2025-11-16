// Function: sub_11CBFA0
// Address: 0x11cbfa0
//
__int64 __fastcall sub_11CBFA0(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4, unsigned __int8 a5)
{
  __int64 v5; // r15
  __int64 *v8; // r13
  unsigned __int64 v9; // rbx
  __int64 v10; // rsi
  char *v11; // r15
  __int64 v12; // r14
  __int64 *v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int8 *v15; // rdx
  unsigned __int8 *v16; // r14
  _QWORD *v17; // rdi
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned int v23; // ecx
  int *v24; // rdx
  int v25; // esi
  int v26; // edx
  int v27; // r9d
  __int64 v28; // [rsp+10h] [rbp-90h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  _QWORD v33[2]; // [rsp+30h] [rbp-70h] BYREF
  char *v34; // [rsp+40h] [rbp-60h] BYREF
  __int64 v35; // [rsp+48h] [rbp-58h]
  _QWORD v36[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v37; // [rsp+60h] [rbp-40h]

  v5 = 0;
  v8 = (__int64 *)sub_AA4B30(*(_QWORD *)(a2 + 48));
  if ( !sub_11C99B0(v8, a3, a4) )
    return v5;
  v9 = a3[((unsigned __int64)a4 >> 6) + 1] & (1LL << a4);
  if ( v9 )
  {
    v9 = 0;
    v11 = 0;
    goto LABEL_7;
  }
  v10 = *a3;
  if ( (((int)*(unsigned __int8 *)(*a3 + (a4 >> 2)) >> (2 * (a4 & 3))) & 3) != 0 )
  {
    if ( (((int)*(unsigned __int8 *)(*a3 + (a4 >> 2)) >> (2 * (a4 & 3))) & 3) == 3 )
    {
      v11 = (&off_4977320)[2 * a4];
      v9 = qword_4977328[2 * a4];
      goto LABEL_7;
    }
    v21 = *(unsigned int *)(v10 + 160);
    v22 = *(_QWORD *)(v10 + 144);
    if ( (_DWORD)v21 )
    {
      v23 = (v21 - 1) & (37 * a4);
      v24 = (int *)(v22 + 40LL * v23);
      v25 = *v24;
      if ( a4 == *v24 )
      {
LABEL_12:
        v11 = (char *)*((_QWORD *)v24 + 1);
        v9 = *((_QWORD *)v24 + 2);
        goto LABEL_7;
      }
      v26 = 1;
      while ( v25 != -1 )
      {
        v27 = v26 + 1;
        v23 = (v21 - 1) & (v26 + v23);
        v24 = (int *)(v22 + 40LL * v23);
        v25 = *v24;
        if ( a4 == *v24 )
          goto LABEL_12;
        v26 = v27;
      }
    }
    v24 = (int *)(v22 + 40 * v21);
    goto LABEL_12;
  }
  v11 = 0;
LABEL_7:
  v12 = sub_BCB2B0(*(_QWORD **)(a2 + 72));
  v28 = *(_QWORD *)(a1 + 8);
  v13 = (__int64 *)sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
  v36[1] = v12;
  v34 = (char *)v36;
  v36[0] = v28;
  v35 = 0x200000002LL;
  v14 = sub_BCF480(v13, v36, 2, 0);
  v29 = sub_BA8C10((__int64)v8, (__int64)v11, v9, v14, 0);
  v16 = v15;
  sub_11C9500((__int64)v8, (__int64)v11, v9, a3);
  v17 = *(_QWORD **)(a2 + 72);
  v34 = v11;
  v37 = 261;
  v35 = v9;
  v33[0] = a1;
  v18 = sub_BCB2B0(v17);
  v33[1] = sub_ACD640(v18, a5, 0);
  v5 = sub_921880((unsigned int **)a2, v29, (int)v16, (int)v33, 2, (__int64)&v34, 0);
  v19 = sub_BD3990(v16, v29);
  if ( !*v19 )
    *(_WORD *)(v5 + 2) = *(_WORD *)(v5 + 2) & 0xF003 | (4 * ((*((_WORD *)v19 + 1) >> 4) & 0x3FF));
  return v5;
}
