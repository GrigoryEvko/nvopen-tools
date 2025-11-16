// Function: sub_318C6D0
// Address: 0x318c6d0
//
_QWORD *__fastcall sub_318C6D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned int v10; // r13d
  __int64 v12; // rax
  unsigned __int8 *v13; // r10
  __int64 v14; // r15
  unsigned __int8 *v15; // rcx
  char *v16; // r12
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, _QWORD, unsigned __int8 *); // rax
  __int64 v23; // rax
  _QWORD **v24; // rdx
  int v25; // esi
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r10
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // rax
  unsigned __int8 *v35; // [rsp+0h] [rbp-80h]
  unsigned __int8 *v36; // [rsp+0h] [rbp-80h]
  unsigned __int8 *v37; // [rsp+0h] [rbp-80h]
  unsigned __int8 *v39; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v40; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v41; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  _DWORD v43[8]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v44; // [rsp+40h] [rbp-40h]

  v10 = a1;
  v12 = sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v13 = *(unsigned __int8 **)(a2 + 16);
  v14 = v12;
  v15 = *(unsigned __int8 **)(a3 + 16);
  if ( (unsigned int)a1 <= 0xF )
  {
    v43[1] = 0;
    v16 = (char *)sub_B35C90(v12, a1, (__int64)v13, (__int64)v15, a5, 0, v43[0], 0);
    goto LABEL_3;
  }
  v21 = *(_QWORD *)(v12 + 80);
  v22 = *(__int64 (__fastcall **)(__int64, _QWORD, unsigned __int8 *))(*(_QWORD *)v21 + 56LL);
  if ( (char *)v22 != (char *)sub_928890 )
  {
    v37 = *(unsigned __int8 **)(a3 + 16);
    v41 = *(unsigned __int8 **)(a2 + 16);
    v34 = v22(v21, v10, v13);
    v15 = v37;
    v13 = v41;
    v16 = (char *)v34;
LABEL_9:
    if ( v16 )
      goto LABEL_3;
    goto LABEL_10;
  }
  if ( *v13 <= 0x15u && *v15 <= 0x15u )
  {
    v35 = *(unsigned __int8 **)(a3 + 16);
    v39 = *(unsigned __int8 **)(a2 + 16);
    v23 = sub_AAB310(v10, v39, v35);
    v13 = v39;
    v15 = v35;
    v16 = (char *)v23;
    goto LABEL_9;
  }
LABEL_10:
  v36 = v13;
  v44 = 257;
  v40 = v15;
  v16 = (char *)sub_BD2C40(72, unk_3F10FD0);
  if ( v16 )
  {
    v24 = (_QWORD **)*((_QWORD *)v36 + 1);
    v25 = *((unsigned __int8 *)v24 + 8);
    if ( (unsigned int)(v25 - 17) > 1 )
    {
      v27 = sub_BCB2A0(*v24);
      v29 = (__int64)v36;
      v28 = (__int64)v40;
    }
    else
    {
      BYTE4(v42) = (_BYTE)v25 == 18;
      LODWORD(v42) = *((_DWORD *)v24 + 8);
      v26 = (__int64 *)sub_BCB2A0(*v24);
      v27 = sub_BCE1B0(v26, v42);
      v28 = (__int64)v40;
      v29 = (__int64)v36;
    }
    sub_B523C0((__int64)v16, v27, 53, v10, v29, v28, (__int64)v43, 0, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD, char *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v14 + 88) + 16LL))(
    *(_QWORD *)(v14 + 88),
    v16,
    a5,
    *(_QWORD *)(v14 + 56),
    *(_QWORD *)(v14 + 64));
  v30 = *(_QWORD *)v14;
  v31 = *(_QWORD *)v14 + 16LL * *(unsigned int *)(v14 + 8);
  if ( *(_QWORD *)v14 != v31 )
  {
    do
    {
      v32 = *(_QWORD *)(v30 + 8);
      v33 = *(_DWORD *)v30;
      v30 += 16;
      sub_B99FD0((__int64)v16, v33, v32);
    }
    while ( v31 != v30 );
    v17 = *v16;
    v18 = (__int64)v16;
    v19 = a4;
    if ( (unsigned __int8)*v16 <= 0x15u )
      return (_QWORD *)sub_31892C0(v19, v18);
    goto LABEL_17;
  }
LABEL_3:
  v17 = *v16;
  v18 = (__int64)v16;
  v19 = a4;
  if ( (unsigned __int8)*v16 <= 0x15u )
    return (_QWORD *)sub_31892C0(v19, v18);
LABEL_17:
  if ( v17 == 82 )
    return sub_318A070(v19, v18);
  else
    return sub_318A0F0(v19, v18);
}
