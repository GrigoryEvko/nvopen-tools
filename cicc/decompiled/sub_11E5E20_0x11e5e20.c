// Function: sub_11E5E20
// Address: 0x11e5e20
//
__int64 __fastcall sub_11E5E20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int8 *v11; // r14
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v13; // r15
  __int64 **v14; // r14
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v17; // r13
  _QWORD **v19; // rdx
  int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // rsi
  unsigned int *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned int v26; // esi
  _QWORD *v27; // rax
  unsigned int *v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // [rsp+18h] [rbp-98h]
  _QWORD v33[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v34; // [rsp+40h] [rbp-70h]
  _QWORD v35[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v36; // [rsp+70h] [rbp-40h]

  v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *(_QWORD *)(v5 + 8);
  v35[0] = "isdigittmp";
  v36 = 259;
  v7 = (_BYTE *)sub_AD64C0(v6, 48, 0);
  v8 = sub_929DE0((unsigned int **)a3, (_BYTE *)v5, v7, (__int64)v35, 0, 0);
  v33[0] = "isdigit";
  v34 = 259;
  v9 = sub_AD64C0(v6, 10, 0);
  v10 = *(_QWORD *)(a3 + 80);
  v11 = (unsigned __int8 *)v9;
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v10 + 56LL);
  if ( v12 != sub_928890 )
  {
    v13 = v12(v10, 36u, (_BYTE *)v8, v11);
LABEL_5:
    if ( v13 )
      goto LABEL_6;
    goto LABEL_13;
  }
  if ( *(_BYTE *)v8 <= 0x15u && *v11 <= 0x15u )
  {
    v13 = sub_AAB310(0x24u, (unsigned __int8 *)v8, v11);
    goto LABEL_5;
  }
LABEL_13:
  v36 = 257;
  v13 = (__int64)sub_BD2C40(72, unk_3F10FD0);
  if ( v13 )
  {
    v19 = *(_QWORD ***)(v8 + 8);
    v20 = *((unsigned __int8 *)v19 + 8);
    if ( (unsigned int)(v20 - 17) > 1 )
    {
      v22 = sub_BCB2A0(*v19);
    }
    else
    {
      BYTE4(v32) = (_BYTE)v20 == 18;
      LODWORD(v32) = *((_DWORD *)v19 + 8);
      v21 = (__int64 *)sub_BCB2A0(*v19);
      v22 = sub_BCE1B0(v21, v32);
    }
    sub_B523C0(v13, v22, 53, 36, v8, (__int64)v11, (__int64)v35, 0, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v13,
    v33,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v23 = *(unsigned int **)a3;
  v24 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v24 )
  {
    do
    {
      v25 = *((_QWORD *)v23 + 1);
      v26 = *v23;
      v23 += 4;
      sub_B99FD0(v13, v26, v25);
    }
    while ( (unsigned int *)v24 != v23 );
  }
LABEL_6:
  v14 = *(__int64 ***)(a2 + 8);
  v34 = 257;
  if ( v14 == *(__int64 ***)(v13 + 8) )
    return v13;
  v15 = *(_QWORD *)(a3 + 80);
  v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v15 + 120LL);
  if ( v16 != sub_920130 )
  {
    v17 = v16(v15, 39u, (_BYTE *)v13, (__int64)v14);
    goto LABEL_11;
  }
  if ( *(_BYTE *)v13 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v17 = sub_ADAB70(39, v13, v14, 0);
    else
      v17 = sub_AA93C0(0x27u, v13, (__int64)v14);
LABEL_11:
    if ( v17 )
      return v17;
  }
  v36 = 257;
  v27 = sub_BD2C40(72, unk_3F10A14);
  v17 = (__int64)v27;
  if ( v27 )
    sub_B515B0((__int64)v27, v13, (__int64)v14, (__int64)v35, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v17,
    v33,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v28 = *(unsigned int **)a3;
  v29 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  while ( (unsigned int *)v29 != v28 )
  {
    v30 = *((_QWORD *)v28 + 1);
    v31 = *v28;
    v28 += 4;
    sub_B99FD0(v17, v31, v30);
  }
  return v17;
}
