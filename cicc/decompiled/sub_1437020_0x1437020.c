// Function: sub_1437020
// Address: 0x1437020
//
__int64 __fastcall sub_1437020(__int64 a1, __int64 a2, __int64 a3, char *a4)
{
  unsigned int v5; // r12d
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r8
  _QWORD *v12; // rbx
  _BYTE *v13; // rcx
  int v14; // eax
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rdx
  unsigned __int8 v20; // al
  _QWORD *v21; // rax
  __int64 v22; // r10
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r11
  char v26; // di
  unsigned int v27; // r8d
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r10
  __int64 v33; // rsi
  int v34; // edi
  __int64 **v35; // rax
  __int64 v36; // [rsp+8h] [rbp-2C8h]
  __int64 v37; // [rsp+10h] [rbp-2C0h]
  __int64 v38; // [rsp+18h] [rbp-2B8h]
  __int64 v39; // [rsp+20h] [rbp-2B0h]
  char v41; // [rsp+47h] [rbp-289h]
  _BYTE *v42; // [rsp+48h] [rbp-288h]
  __int64 v43[6]; // [rsp+50h] [rbp-280h] BYREF
  _BYTE *v44; // [rsp+80h] [rbp-250h] BYREF
  __int64 v45; // [rsp+88h] [rbp-248h]
  _BYTE v46[64]; // [rsp+90h] [rbp-240h] BYREF
  _BYTE v47[512]; // [rsp+D0h] [rbp-200h] BYREF

  if ( **(_QWORD **)(a3 + 32) == *(_QWORD *)(a1 + 40) )
  {
    v5 = 1;
    if ( a4[1] )
      LOBYTE(v5) = a1 == sub_157ED60();
    return v5;
  }
  v5 = 0;
  v41 = *a4;
  if ( *a4 )
    return v5;
  if ( sub_13FCB50(a3) )
  {
    v8 = sub_13FCB50(a3);
    v41 = sub_15CC8F0(a2, *(_QWORD *)(a1 + 40), v8, v9, v10);
  }
  v44 = v46;
  v45 = 0x800000000LL;
  sub_13F9EC0(a3, (__int64)&v44);
  v12 = v44;
  v13 = &v44[8 * (unsigned int)v45];
  v14 = v45;
  v42 = v13;
  if ( v44 == v13 )
  {
LABEL_19:
    LOBYTE(v5) = v14 != 0;
    goto LABEL_22;
  }
  while ( 1 )
  {
    v15 = *v12;
    v5 = sub_15CC8F0(a2, *(_QWORD *)(a1 + 40), *v12, v13, v11);
    if ( (_BYTE)v5 )
      goto LABEL_17;
    if ( !v41 )
      goto LABEL_21;
    v16 = sub_157F0B0(v15);
    if ( !v16 )
      goto LABEL_21;
    v17 = sub_157EBA0(v16);
    v18 = v17;
    if ( *(_BYTE *)(v17 + 16) != 26 || (*(_DWORD *)(v17 + 20) & 0xFFFFFFF) != 3 )
      goto LABEL_21;
    v19 = *(_QWORD *)(v17 - 72);
    v20 = *(_BYTE *)(v19 + 16);
    if ( v20 != 13 )
      break;
    v21 = *(_QWORD **)(v19 + 24);
    if ( *(_DWORD *)(v19 + 32) > 0x40u )
      v21 = (_QWORD *)*v21;
    LOBYTE(v5) = *(_QWORD *)(v18 - 24LL * (v21 != 0) - 24) == v15;
LABEL_16:
    if ( !(_BYTE)v5 )
      goto LABEL_21;
LABEL_17:
    if ( v42 == (_BYTE *)++v12 )
    {
      v14 = v45;
      v42 = v44;
      goto LABEL_19;
    }
  }
  if ( v20 <= 0x17u )
    goto LABEL_21;
  if ( (unsigned __int8)(v20 - 75) > 1u )
    goto LABEL_21;
  v22 = *(_QWORD *)(v19 - 48);
  if ( *(_BYTE *)(v22 + 16) != 77 || *(_QWORD *)(v22 + 40) != **(_QWORD **)(a3 + 32) )
    goto LABEL_21;
  v36 = *(_QWORD *)(v19 - 48);
  v37 = v18;
  v38 = v19;
  v39 = *(_QWORD *)(v19 - 24);
  v23 = sub_157EB90(v15);
  v24 = sub_1632FA0(v23);
  sub_12BDBC0((__int64)v47, v24);
  v25 = sub_13FC520(a3);
  v26 = *(_BYTE *)(v36 + 23) & 0x40;
  v27 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
  if ( v27 )
  {
    v28 = 24LL * *(unsigned int *)(v36 + 56) + 8;
    v29 = 0;
    while ( 1 )
    {
      v30 = v36 - 24LL * v27;
      if ( v26 )
        v30 = *(_QWORD *)(v36 - 8);
      if ( v25 == *(_QWORD *)(v30 + v28) )
        break;
      ++v29;
      v28 += 8;
      if ( v27 == (_DWORD)v29 )
        goto LABEL_43;
    }
    v31 = 24 * v29;
  }
  else
  {
LABEL_43:
    v31 = 0x17FFFFFFE8LL;
  }
  if ( v26 )
    v32 = *(_QWORD *)(v36 - 8);
  else
    v32 = v36 - 24LL * v27;
  v33 = *(_QWORD *)(v32 + v31);
  v43[4] = v37;
  v43[2] = a2;
  v43[3] = 0;
  v34 = *(unsigned __int16 *)(v38 + 18);
  v43[0] = (__int64)v47;
  v43[1] = 0;
  v35 = sub_13E1240(v34 & 0xFFFF7FFF, v33, v39, v43);
  if ( v35 && *((_BYTE *)v35 + 16) <= 0x10u )
  {
    if ( v15 == *(_QWORD *)(v37 - 24) )
      v5 = sub_1595F50(v35);
    else
      v5 = sub_1596070(v35);
    sub_15A93E0(v47);
    goto LABEL_16;
  }
  sub_15A93E0(v47);
LABEL_21:
  v42 = v44;
LABEL_22:
  if ( v42 != v46 )
    _libc_free((unsigned __int64)v42);
  return v5;
}
