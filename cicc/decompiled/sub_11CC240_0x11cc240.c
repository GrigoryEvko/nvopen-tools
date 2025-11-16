// Function: sub_11CC240
// Address: 0x11cc240
//
__int64 __fastcall sub_11CC240(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, unsigned int a5, unsigned __int8 a6)
{
  __int64 v6; // r15
  __int64 *v9; // r13
  unsigned __int64 v10; // rbx
  __int64 v11; // rsi
  char *v12; // r15
  __int64 v13; // r14
  __int64 *v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int8 *v16; // rdx
  unsigned __int8 *v17; // r14
  _QWORD *v18; // rdi
  __int64 v19; // rax
  unsigned __int8 *v20; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned int v24; // ecx
  int *v25; // rdx
  int v26; // esi
  int v27; // edx
  int v28; // r9d
  __int64 v29; // [rsp+0h] [rbp-B0h]
  __int64 v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+8h] [rbp-A8h]
  _QWORD v36[4]; // [rsp+30h] [rbp-80h] BYREF
  char *v37; // [rsp+50h] [rbp-60h] BYREF
  __int64 v38; // [rsp+58h] [rbp-58h]
  _QWORD v39[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v40; // [rsp+70h] [rbp-40h]

  v6 = 0;
  v9 = (__int64 *)sub_AA4B30(*(_QWORD *)(a3 + 48));
  if ( !sub_11C99B0(v9, a4, a5) )
    return v6;
  v10 = a4[((unsigned __int64)a5 >> 6) + 1] & (1LL << a5);
  if ( v10 )
  {
    v10 = 0;
    v12 = 0;
    goto LABEL_7;
  }
  v11 = *a4;
  if ( (((int)*(unsigned __int8 *)(*a4 + (a5 >> 2)) >> (2 * (a5 & 3))) & 3) != 0 )
  {
    if ( (((int)*(unsigned __int8 *)(*a4 + (a5 >> 2)) >> (2 * (a5 & 3))) & 3) == 3 )
    {
      v12 = (&off_4977320)[2 * a5];
      v10 = qword_4977328[2 * a5];
      goto LABEL_7;
    }
    v22 = *(unsigned int *)(v11 + 160);
    v23 = *(_QWORD *)(v11 + 144);
    if ( (_DWORD)v22 )
    {
      v24 = (v22 - 1) & (37 * a5);
      v25 = (int *)(v23 + 40LL * v24);
      v26 = *v25;
      if ( a5 == *v25 )
      {
LABEL_12:
        v12 = (char *)*((_QWORD *)v25 + 1);
        v10 = *((_QWORD *)v25 + 2);
        goto LABEL_7;
      }
      v27 = 1;
      while ( v26 != -1 )
      {
        v28 = v27 + 1;
        v24 = (v22 - 1) & (v27 + v24);
        v25 = (int *)(v23 + 40LL * v24);
        v26 = *v25;
        if ( a5 == *v25 )
          goto LABEL_12;
        v27 = v28;
      }
    }
    v25 = (int *)(v23 + 40 * v22);
    goto LABEL_12;
  }
  v12 = 0;
LABEL_7:
  v13 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
  v29 = *(_QWORD *)(a2 + 8);
  v30 = *(_QWORD *)(a1 + 8);
  v14 = (__int64 *)sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
  v37 = (char *)v39;
  v39[0] = v30;
  v39[1] = v29;
  v40 = v13;
  v38 = 0x300000003LL;
  v15 = sub_BCF480(v14, v39, 3, 0);
  v31 = sub_BA8C10((__int64)v9, (__int64)v12, v10, v15, 0);
  v17 = v16;
  sub_11C9500((__int64)v9, (__int64)v12, v10, a4);
  v18 = *(_QWORD **)(a3 + 72);
  v37 = v12;
  LOWORD(v40) = 261;
  v38 = v10;
  v36[0] = a1;
  v36[1] = a2;
  v19 = sub_BCB2B0(v18);
  v36[2] = sub_ACD640(v19, a6, 0);
  v6 = sub_921880((unsigned int **)a3, v31, (int)v17, (int)v36, 3, (__int64)&v37, 0);
  v20 = sub_BD3990(v17, v31);
  if ( !*v20 )
    *(_WORD *)(v6 + 2) = *(_WORD *)(v6 + 2) & 0xF003 | (4 * ((*((_WORD *)v20 + 1) >> 4) & 0x3FF));
  return v6;
}
