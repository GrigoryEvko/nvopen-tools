// Function: sub_11E6190
// Address: 0x11e6190
//
__int64 __fastcall sub_11E6190(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int8 *v9; // rbx
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v11; // r15
  __int64 **v12; // r14
  __int64 v13; // rdi
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v15; // r13
  _QWORD **v17; // rdx
  int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rdx
  unsigned int v24; // esi
  _QWORD *v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // [rsp+8h] [rbp-98h]
  _QWORD v31[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v32; // [rsp+30h] [rbp-70h]
  _BYTE v33[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *(_QWORD *)(v5 + 8);
  v31[0] = "isascii";
  v32 = 259;
  v7 = sub_AD64C0(v6, 128, 0);
  v8 = a3[10];
  v9 = (unsigned __int8 *)v7;
  v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v8 + 56LL);
  if ( v10 != sub_928890 )
  {
    v11 = v10(v8, 36u, (_BYTE *)v5, v9);
LABEL_5:
    if ( v11 )
      goto LABEL_6;
    goto LABEL_13;
  }
  if ( *(_BYTE *)v5 <= 0x15u && *v9 <= 0x15u )
  {
    v11 = sub_AAB310(0x24u, (unsigned __int8 *)v5, v9);
    goto LABEL_5;
  }
LABEL_13:
  v34 = 257;
  v11 = (__int64)sub_BD2C40(72, unk_3F10FD0);
  if ( v11 )
  {
    v17 = *(_QWORD ***)(v5 + 8);
    v18 = *((unsigned __int8 *)v17 + 8);
    if ( (unsigned int)(v18 - 17) > 1 )
    {
      v20 = sub_BCB2A0(*v17);
    }
    else
    {
      BYTE4(v30) = (_BYTE)v18 == 18;
      LODWORD(v30) = *((_DWORD *)v17 + 8);
      v19 = (__int64 *)sub_BCB2A0(*v17);
      v20 = sub_BCE1B0(v19, v30);
    }
    sub_B523C0(v11, v20, 53, 36, v5, (__int64)v9, (__int64)v33, 0, 0, 0);
  }
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
    a3[11],
    v11,
    v31,
    a3[7],
    a3[8]);
  v21 = *a3;
  v22 = *a3 + 16LL * *((unsigned int *)a3 + 2);
  if ( *a3 != v22 )
  {
    do
    {
      v23 = *(_QWORD *)(v21 + 8);
      v24 = *(_DWORD *)v21;
      v21 += 16;
      sub_B99FD0(v11, v24, v23);
    }
    while ( v22 != v21 );
  }
LABEL_6:
  v12 = *(__int64 ***)(a2 + 8);
  v32 = 257;
  if ( v12 == *(__int64 ***)(v11 + 8) )
    return v11;
  v13 = a3[10];
  v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v13 + 120LL);
  if ( v14 != sub_920130 )
  {
    v15 = v14(v13, 39u, (_BYTE *)v11, (__int64)v12);
    goto LABEL_11;
  }
  if ( *(_BYTE *)v11 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v15 = sub_ADAB70(39, v11, v12, 0);
    else
      v15 = sub_AA93C0(0x27u, v11, (__int64)v12);
LABEL_11:
    if ( v15 )
      return v15;
  }
  v34 = 257;
  v25 = sub_BD2C40(72, unk_3F10A14);
  v15 = (__int64)v25;
  if ( v25 )
    sub_B515B0((__int64)v25, v11, (__int64)v12, (__int64)v33, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
    a3[11],
    v15,
    v31,
    a3[7],
    a3[8]);
  v26 = *a3;
  v27 = *a3 + 16LL * *((unsigned int *)a3 + 2);
  while ( v27 != v26 )
  {
    v28 = *(_QWORD *)(v26 + 8);
    v29 = *(_DWORD *)v26;
    v26 += 16;
    sub_B99FD0(v15, v29, v28);
  }
  return v15;
}
