// Function: sub_29DE300
// Address: 0x29de300
//
void __fastcall sub_29DE300(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r15
  unsigned __int8 *v3; // r14
  __int64 (__fastcall *v4)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v5; // r13
  __int64 v6; // rax
  char *v7; // r12
  char *v8; // r14
  __int64 v9; // rdx
  unsigned int v10; // esi
  __int64 v11; // rax
  unsigned __int8 *v12; // r14
  __int64 (__fastcall *v13)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  char *v18; // r12
  char *v19; // r14
  __int64 v20; // rdx
  unsigned int v21; // esi
  unsigned __int8 **v22; // [rsp+0h] [rbp-140h] BYREF
  unsigned __int8 **v23; // [rsp+8h] [rbp-138h] BYREF
  __int64 v24; // [rsp+10h] [rbp-130h] BYREF
  __int64 v25; // [rsp+18h] [rbp-128h] BYREF
  _BYTE v26[32]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v27; // [rsp+40h] [rbp-100h]
  _BYTE v28[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v29; // [rsp+70h] [rbp-D0h]
  char *v30; // [rsp+80h] [rbp-C0h] BYREF
  int v31; // [rsp+88h] [rbp-B8h]
  char v32; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+B8h] [rbp-88h]
  __int64 v34; // [rsp+C0h] [rbp-80h]
  __int64 v35; // [rsp+D0h] [rbp-70h]
  __int64 v36; // [rsp+D8h] [rbp-68h]
  void *v37; // [rsp+100h] [rbp-40h]

  sub_D22920(a1, (__int64 *)&v22, (__int64 *)&v23, &v24, &v25);
  if ( v22 )
  {
    sub_23D0AB0((__int64)&v30, a1, 0, 0, 0);
    v2 = (__int64)v22;
    v27 = 257;
    v3 = *v22;
    v4 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v35 + 16LL);
    if ( v4 == sub_9202E0 )
    {
      if ( *a2 > 0x15u || *v3 > 0x15u )
        goto LABEL_11;
      if ( (unsigned __int8)sub_AC47B0(28) )
        v5 = sub_AD5570(28, (__int64)a2, v3, 0, 0);
      else
        v5 = sub_AABE40(0x1Cu, a2, v3);
    }
    else
    {
      v5 = v4(v35, 28u, a2, *v22);
    }
    if ( v5 )
    {
      if ( !*(_QWORD *)v2 || (v6 = *(_QWORD *)(v2 + 8), (**(_QWORD **)(v2 + 16) = v6) == 0) )
      {
        *(_QWORD *)v2 = v5;
        goto LABEL_17;
      }
      goto LABEL_15;
    }
LABEL_11:
    v29 = 257;
    v5 = sub_B504D0(28, (__int64)a2, (__int64)v3, (__int64)v28, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v36 + 16LL))(
      v36,
      v5,
      v26,
      v33,
      v34);
    v7 = v30;
    v8 = &v30[16 * v31];
    if ( v30 != v8 )
    {
      do
      {
        v9 = *((_QWORD *)v7 + 1);
        v10 = *(_DWORD *)v7;
        v7 += 16;
        sub_B99FD0(v5, v10, v9);
      }
      while ( v8 != v7 );
    }
    if ( !*(_QWORD *)v2 || (v6 = *(_QWORD *)(v2 + 8), (**(_QWORD **)(v2 + 16) = v6) == 0) )
    {
LABEL_16:
      *(_QWORD *)v2 = v5;
      if ( !v5 )
      {
LABEL_20:
        sub_B444E0(*(_QWORD **)(a1 - 96), a1 + 24, 0);
        goto LABEL_21;
      }
LABEL_17:
      v11 = *(_QWORD *)(v5 + 16);
      *(_QWORD *)(v2 + 8) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v2 + 8;
      *(_QWORD *)(v2 + 16) = v5 + 16;
      *(_QWORD *)(v5 + 16) = v2;
      goto LABEL_20;
    }
LABEL_15:
    *(_QWORD *)(v6 + 16) = *(_QWORD *)(v2 + 16);
    goto LABEL_16;
  }
  sub_23D0AB0((__int64)&v30, a1, 0, 0, 0);
  v27 = 257;
  v12 = *v23;
  v13 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v35 + 16LL);
  if ( v13 == sub_9202E0 )
  {
    if ( *a2 > 0x15u || *v12 > 0x15u )
    {
      v15 = a1 - 96;
LABEL_39:
      v29 = 257;
      v14 = sub_B504D0(28, (__int64)a2, (__int64)v12, (__int64)v28, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v36 + 16LL))(
        v36,
        v14,
        v26,
        v33,
        v34);
      v18 = v30;
      v19 = &v30[16 * v31];
      if ( v30 != v19 )
      {
        do
        {
          v20 = *((_QWORD *)v18 + 1);
          v21 = *(_DWORD *)v18;
          v18 += 16;
          sub_B99FD0(v14, v21, v20);
        }
        while ( v19 != v18 );
      }
      if ( !*(_QWORD *)(a1 - 96) || (v16 = *(_QWORD *)(a1 - 88), (**(_QWORD **)(a1 - 80) = v16) == 0) )
      {
LABEL_33:
        *(_QWORD *)(a1 - 96) = v14;
        if ( !v14 )
          goto LABEL_21;
        goto LABEL_34;
      }
LABEL_32:
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(a1 - 80);
      goto LABEL_33;
    }
    if ( (unsigned __int8)sub_AC47B0(28) )
      v14 = sub_AD5570(28, (__int64)a2, v12, 0, 0);
    else
      v14 = sub_AABE40(0x1Cu, a2, v12);
  }
  else
  {
    v14 = v13(v35, 28u, a2, *v23);
  }
  v15 = a1 - 96;
  if ( !v14 )
    goto LABEL_39;
  if ( *(_QWORD *)(a1 - 96) )
  {
    v16 = *(_QWORD *)(a1 - 88);
    **(_QWORD **)(a1 - 80) = v16;
    if ( v16 )
      goto LABEL_32;
  }
  *(_QWORD *)(a1 - 96) = v14;
LABEL_34:
  v17 = *(_QWORD *)(v14 + 16);
  *(_QWORD *)(a1 - 88) = v17;
  if ( v17 )
    *(_QWORD *)(v17 + 16) = a1 - 88;
  *(_QWORD *)(a1 - 80) = v14 + 16;
  *(_QWORD *)(v14 + 16) = v15;
LABEL_21:
  nullsub_61();
  v37 = &unk_49DA100;
  nullsub_63();
  if ( v30 != &v32 )
    _libc_free((unsigned __int64)v30);
}
