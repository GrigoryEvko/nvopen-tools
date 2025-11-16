// Function: sub_2F3AD20
// Address: 0x2f3ad20
//
__int64 __fastcall sub_2F3AD20(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 (*v4)(); // rax
  void (__fastcall *v5)(__int64, __int64); // rax
  _QWORD *v6; // r12
  __int64 v7; // r15
  __int64 v8; // r15
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 (*v11)(void); // rdx
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r12
  unsigned int v19; // ebx
  unsigned int v20; // r14d
  _QWORD *v21; // r12
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // rbx
  int v24; // edx
  __int64 i; // rbx
  char v26; // al
  unsigned int v27; // ebx
  int v29; // ebx
  __int64 (*v30)(); // rdx
  int v31; // eax
  unsigned __int64 v33; // [rsp+20h] [rbp-EB0h]
  unsigned int v34; // [rsp+2Ch] [rbp-EA4h]
  unsigned __int64 v35[2]; // [rsp+30h] [rbp-EA0h] BYREF
  _BYTE v36[32]; // [rsp+40h] [rbp-E90h] BYREF
  _QWORD v37[432]; // [rsp+60h] [rbp-E70h] BYREF
  _QWORD v38[14]; // [rsp+DE0h] [rbp-F0h] BYREF
  __int64 v39; // [rsp+E50h] [rbp-80h]
  __int64 v40; // [rsp+E58h] [rbp-78h]
  __int64 v41; // [rsp+E60h] [rbp-70h]
  __int64 v42; // [rsp+E68h] [rbp-68h]
  __int64 v43; // [rsp+E70h] [rbp-60h]
  _QWORD v44[3]; // [rsp+E78h] [rbp-58h] BYREF
  unsigned int v45; // [rsp+E90h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 16);
  if ( !HIWORD(dword_502328C) )
  {
    v29 = *(_DWORD *)(a1[3] + 648LL);
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 296LL))(v2) )
    {
      v30 = *(__int64 (**)())(*(_QWORD *)v2 + 392LL);
      v31 = 2;
      if ( v30 != sub_2F39200 )
        v31 = ((__int64 (__fastcall *)(__int64))v30)(v2);
      if ( v29 >= v31 )
        goto LABEL_3;
    }
    return 0;
  }
  if ( !(_BYTE)qword_5023308 )
    return 0;
LABEL_3:
  LODWORD(v3) = 0;
  v4 = *(__int64 (**)())(*(_QWORD *)v2 + 352LL);
  if ( v4 != sub_2F391D0 )
    LODWORD(v3) = ((__int64 (__fastcall *)(__int64))v4)(v2);
  if ( word_502318E )
  {
    LODWORD(v3) = 2;
    if ( sub_2241AC0((__int64)&unk_5023208, "all") )
      v3 = sub_2241AC0((__int64)&unk_5023208, "critical") == 0;
  }
  v35[0] = (unsigned __int64)v36;
  v35[1] = 0x400000000LL;
  v5 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 360LL);
  if ( v5 != sub_2F391E0 )
    v5(v2, (__int64)v35);
  v6 = a1 + 4;
  sub_2F5FFA0(a1 + 4, a2);
  v7 = a1[2];
  sub_2F91670(v37, a2, a1[1], 0);
  memset(&v38[1], 0, 64);
  v40 = v7;
  v8 = 0;
  v38[0] = &unk_4A38790;
  v38[9] = v38;
  v9 = *(__int64 **)(a2 + 16);
  memset(&v38[10], 0, 24);
  v41 = 0;
  v42 = 0;
  v43 = 0;
  memset(v44, 0, sizeof(v44));
  v45 = 0;
  v37[0] = off_4A2A8D8;
  v10 = *v9;
  v11 = *(__int64 (**)(void))(*v9 + 216);
  if ( v11 != sub_2F391C0 )
  {
    v8 = v11();
    v10 = **(_QWORD **)(a2 + 16);
  }
  v12 = *(__int64 (**)())(v10 + 128);
  if ( v12 == sub_2DAC790 )
    BUG();
  v13 = v12();
  v38[13] = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v13 + 1048LL))(v13, v8, v37);
  v14 = *(_QWORD *)(a2 + 16);
  v15 = *(void (**)())(*(_QWORD *)v14 + 368LL);
  if ( v15 != nullsub_1638 )
    ((void (__fastcall *)(__int64, _QWORD *))v15)(v14, v44);
  if ( (_DWORD)v3 == 2 )
  {
    v16 = sub_34B3E50(a2, v6, v35);
  }
  else
  {
    v16 = 0;
    if ( (_DWORD)v3 == 1 )
      v16 = sub_34E0000(a2, v6);
  }
  v39 = v16;
  v17 = *(_QWORD *)(a2 + 328);
  if ( a2 + 320 != v17 )
  {
    while ( 1 )
    {
      sub_2F39590((__int64)v37, v17);
      v18 = *(_QWORD *)(v17 + 56);
      if ( v18 != v17 + 48 )
        break;
      v27 = 0;
      sub_2F90C80(v37, v17, v18, *(_QWORD *)(v17 + 56), 0);
LABEL_40:
      if ( v41 != v42 )
        v42 = v41;
      v45 = v27;
      sub_2F3A550((__int64)v37);
      nullsub_1668(v37);
      sub_2F39760((__int64)v37);
      if ( v39 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 40LL))(v39);
      sub_2F90C70(v37);
      sub_2F92740(v37, v17);
      v17 = *(_QWORD *)(v17 + 8);
      if ( a2 + 320 == v17 )
        goto LABEL_45;
    }
    v19 = 0;
    do
    {
      v18 = *(_QWORD *)(v18 + 8);
      ++v19;
    }
    while ( v18 != v17 + 48 );
    v33 = v17 + 48;
    v20 = v19;
    v21 = (_QWORD *)(v17 + 48);
    v34 = v19;
    while ( 1 )
    {
      v22 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v22 )
        BUG();
      v23 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
      --v20;
      v24 = *(_DWORD *)(v22 + 44);
      if ( (*(_QWORD *)v22 & 4) != 0 )
      {
        if ( (v24 & 4) != 0 )
          goto LABEL_49;
      }
      else if ( (v24 & 4) != 0 )
      {
        for ( i = *(_QWORD *)v22; ; i = *(_QWORD *)v23 )
        {
          v23 = i & 0xFFFFFFFFFFFFFFF8LL;
          v24 = *(_DWORD *)(v23 + 44) & 0xFFFFFF;
          if ( (*(_DWORD *)(v23 + 44) & 4) == 0 )
            break;
        }
      }
      if ( (v24 & 8) != 0 )
      {
        v26 = sub_2E88A90(v23, 128, 1);
        goto LABEL_29;
      }
LABEL_49:
      v26 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v23 + 16) + 24LL) >> 7;
LABEL_29:
      if ( v26
        || (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64))(*(_QWORD *)*a1 + 1016LL))(
             *a1,
             v23,
             v17,
             a2) )
      {
        sub_2F90C80(v37, v17, v21, v33, v34 - v20);
        if ( v41 != v42 )
          v42 = v41;
        v45 = v34;
        sub_2F3A550((__int64)v37);
        nullsub_1668(v37);
        sub_2F39760((__int64)v37);
        if ( v39 )
          (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD, _QWORD))(*(_QWORD *)v39 + 32LL))(
            v39,
            v23,
            v20,
            v45);
        v33 = v23;
        v34 = v20;
      }
      if ( *(_WORD *)(v23 + 68) == 21 )
        v20 -= sub_2E89C40(v23);
      v21 = (_QWORD *)v23;
      if ( *(_QWORD *)(v17 + 56) == v23 )
      {
        v27 = v34;
        sub_2F90C80(v37, v17, v21, v33, v34);
        goto LABEL_40;
      }
    }
  }
LABEL_45:
  sub_2F39BC0(v37);
  if ( (_BYTE *)v35[0] != v36 )
    _libc_free(v35[0]);
  return 1;
}
