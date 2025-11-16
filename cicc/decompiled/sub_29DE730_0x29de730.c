// Function: sub_29DE730
// Address: 0x29de730
//
void __fastcall sub_29DE730(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned __int8 *v5; // r14
  __int64 (__fastcall *v6)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  char *v11; // r12
  char *v12; // r14
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // [rsp+0h] [rbp-140h] BYREF
  unsigned __int8 **v16; // [rsp+8h] [rbp-138h] BYREF
  __int64 v17; // [rsp+10h] [rbp-130h] BYREF
  __int64 v18; // [rsp+18h] [rbp-128h] BYREF
  char v19[32]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v20; // [rsp+40h] [rbp-100h]
  char v21[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v22; // [rsp+70h] [rbp-D0h]
  char *v23; // [rsp+80h] [rbp-C0h] BYREF
  int v24; // [rsp+88h] [rbp-B8h]
  char v25; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v26; // [rsp+B8h] [rbp-88h]
  __int64 v27; // [rsp+C0h] [rbp-80h]
  __int64 v28; // [rsp+D0h] [rbp-70h]
  __int64 v29; // [rsp+D8h] [rbp-68h]
  void *v30; // [rsp+100h] [rbp-40h]

  sub_D22920(a1, &v15, (__int64 *)&v16, &v17, &v18);
  if ( v15 )
  {
    sub_B444E0(*(_QWORD **)(a1 - 96), a1 + 24, 0);
    v2 = (__int64 *)v15;
    if ( *(_QWORD *)v15 )
    {
      v3 = *(_QWORD *)(v15 + 8);
      **(_QWORD **)(v15 + 16) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = v2[2];
    }
    *v2 = a2;
    if ( a2 )
    {
      v4 = *(_QWORD *)(a2 + 16);
      v2[1] = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = v2 + 1;
      v2[2] = a2 + 16;
      *(_QWORD *)(a2 + 16) = v2;
    }
    return;
  }
  sub_23D0AB0((__int64)&v23, a1, 0, 0, 0);
  v20 = 257;
  v5 = *v16;
  v6 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v28 + 16LL);
  if ( v6 == sub_9202E0 )
  {
    if ( *(_BYTE *)a2 > 0x15u || *v5 > 0x15u )
    {
      v8 = a1 - 96;
      goto LABEL_26;
    }
    if ( (unsigned __int8)sub_AC47B0(28) )
      v7 = sub_AD5570(28, a2, v5, 0, 0);
    else
      v7 = sub_AABE40(0x1Cu, (unsigned __int8 *)a2, v5);
  }
  else
  {
    v7 = v6(v28, 28u, (_BYTE *)a2, *v16);
  }
  v8 = a1 - 96;
  if ( v7 )
  {
    if ( !*(_QWORD *)(a1 - 96) || (v9 = *(_QWORD *)(a1 - 88), (**(_QWORD **)(a1 - 80) = v9) == 0) )
    {
      *(_QWORD *)(a1 - 96) = v7;
      goto LABEL_20;
    }
    goto LABEL_18;
  }
LABEL_26:
  v22 = 257;
  v7 = sub_B504D0(28, a2, (__int64)v5, (__int64)v21, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v29 + 16LL))(v29, v7, v19, v26, v27);
  v11 = v23;
  v12 = &v23[16 * v24];
  if ( v23 != v12 )
  {
    do
    {
      v13 = *((_QWORD *)v11 + 1);
      v14 = *(_DWORD *)v11;
      v11 += 16;
      sub_B99FD0(v7, v14, v13);
    }
    while ( v12 != v11 );
  }
  if ( *(_QWORD *)(a1 - 96) )
  {
    v9 = *(_QWORD *)(a1 - 88);
    **(_QWORD **)(a1 - 80) = v9;
    if ( v9 )
LABEL_18:
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(a1 - 80);
  }
  *(_QWORD *)(a1 - 96) = v7;
  if ( v7 )
  {
LABEL_20:
    v10 = *(_QWORD *)(v7 + 16);
    *(_QWORD *)(a1 - 88) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = a1 - 88;
    *(_QWORD *)(a1 - 80) = v7 + 16;
    *(_QWORD *)(v7 + 16) = v8;
  }
  nullsub_61();
  v30 = &unk_49DA100;
  nullsub_63();
  if ( v23 != &v25 )
    _libc_free((unsigned __int64)v23);
}
