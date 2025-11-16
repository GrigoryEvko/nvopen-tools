// Function: sub_11CC530
// Address: 0x11cc530
//
__int64 __fastcall sub_11CC530(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        unsigned int a6,
        unsigned __int8 a7)
{
  __int64 v8; // r14
  __int64 *v10; // r12
  unsigned __int64 v11; // rbx
  __int64 v12; // rsi
  char *v13; // r14
  __int64 v14; // r13
  __int64 *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // r13
  _QWORD *v19; // rdi
  __int64 v20; // rax
  unsigned __int8 *v21; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned int v25; // ecx
  int *v26; // rdx
  int v27; // esi
  int v28; // edx
  int v29; // r9d
  __int64 v30; // [rsp+0h] [rbp-C0h]
  __int64 v31; // [rsp+8h] [rbp-B8h]
  __int64 v32; // [rsp+10h] [rbp-B0h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  _QWORD v38[4]; // [rsp+40h] [rbp-80h] BYREF
  char *v39; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+68h] [rbp-58h]
  _QWORD v41[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v42; // [rsp+80h] [rbp-40h]
  __int64 v43; // [rsp+88h] [rbp-38h]

  v8 = 0;
  v10 = (__int64 *)sub_AA4B30(*(_QWORD *)(a4 + 48));
  if ( !sub_11C99B0(v10, a5, a6) )
    return v8;
  v11 = a5[((unsigned __int64)a6 >> 6) + 1] & (1LL << a6);
  if ( v11 )
  {
    v11 = 0;
    v13 = 0;
    goto LABEL_7;
  }
  v12 = *a5;
  if ( (((int)*(unsigned __int8 *)(*a5 + (a6 >> 2)) >> (2 * (a6 & 3))) & 3) != 0 )
  {
    if ( (((int)*(unsigned __int8 *)(*a5 + (a6 >> 2)) >> (2 * (a6 & 3))) & 3) == 3 )
    {
      v13 = (&off_4977320)[2 * a6];
      v11 = qword_4977328[2 * a6];
      goto LABEL_7;
    }
    v23 = *(unsigned int *)(v12 + 160);
    v24 = *(_QWORD *)(v12 + 144);
    if ( (_DWORD)v23 )
    {
      v25 = (v23 - 1) & (37 * a6);
      v26 = (int *)(v24 + 40LL * v25);
      v27 = *v26;
      if ( a6 == *v26 )
      {
LABEL_12:
        v13 = (char *)*((_QWORD *)v26 + 1);
        v11 = *((_QWORD *)v26 + 2);
        goto LABEL_7;
      }
      v28 = 1;
      while ( v27 != -1 )
      {
        v29 = v28 + 1;
        v25 = (v23 - 1) & (v28 + v25);
        v26 = (int *)(v24 + 40LL * v25);
        v27 = *v26;
        if ( a6 == *v26 )
          goto LABEL_12;
        v28 = v29;
      }
    }
    v26 = (int *)(v24 + 40 * v23);
    goto LABEL_12;
  }
  v13 = 0;
LABEL_7:
  v14 = sub_BCB2B0(*(_QWORD **)(a4 + 72));
  v30 = *(_QWORD *)(a3 + 8);
  v31 = *(_QWORD *)(a2 + 8);
  v32 = *(_QWORD *)(a1 + 8);
  v15 = (__int64 *)sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  v39 = (char *)v41;
  v41[0] = v32;
  v41[1] = v31;
  v42 = v30;
  v43 = v14;
  v40 = 0x400000004LL;
  v16 = sub_BCF480(v15, v41, 4, 0);
  v33 = sub_BA8C10((__int64)v10, (__int64)v13, v11, v16, 0);
  v18 = v17;
  sub_11C9500((__int64)v10, (__int64)v13, v11, a5);
  v19 = *(_QWORD **)(a4 + 72);
  v39 = v13;
  LOWORD(v42) = 261;
  v40 = v11;
  v38[0] = a1;
  v38[1] = a2;
  v38[2] = a3;
  v20 = sub_BCB2B0(v19);
  v38[3] = sub_ACD640(v20, a7, 0);
  v8 = sub_921880((unsigned int **)a4, v33, (int)v18, (int)v38, 4, (__int64)&v39, 0);
  v21 = sub_BD3990(v18, v33);
  if ( !*v21 )
    *(_WORD *)(v8 + 2) = *(_WORD *)(v8 + 2) & 0xF003 | (4 * ((*((_WORD *)v21 + 1) >> 4) & 0x3FF));
  return v8;
}
