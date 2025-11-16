// Function: sub_324B550
// Address: 0x324b550
//
void __fastcall sub_324B550(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  const void *v9; // rax
  size_t v10; // rdx
  __int64 v11; // rdx
  unsigned __int8 *v12; // rsi
  unsigned __int8 *v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // r8
  __int64 v25; // [rsp+8h] [rbp-D8h]
  unsigned __int64 *v26; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+18h] [rbp-C8h]
  _QWORD v28[3]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE *v29; // [rsp+38h] [rbp-A8h]
  _BYTE v30[60]; // [rsp+48h] [rbp-98h] BYREF
  char v31; // [rsp+84h] [rbp-5Ch]
  __int64 **v32; // [rsp+90h] [rbp-50h]

  v3 = a3 - 16;
  v7 = *(_BYTE *)(a3 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL);
    if ( !v8 )
      goto LABEL_6;
  }
  else
  {
    v8 = *(_QWORD *)(a3 - 8LL * ((v7 >> 2) & 0xF));
    if ( !v8 )
      goto LABEL_19;
  }
  v9 = (const void *)sub_B91420(v8);
  if ( v10 )
    sub_324AD70(a1, a2, 3, v9, v10);
  v7 = *(_BYTE *)(a3 - 16);
  if ( (v7 & 2) != 0 )
  {
LABEL_6:
    v11 = *(_QWORD *)(a3 - 32);
    v12 = *(unsigned __int8 **)(v11 + 24);
    if ( v12 )
      goto LABEL_7;
    goto LABEL_20;
  }
LABEL_19:
  v11 = v3 - 8LL * ((v7 >> 2) & 0xF);
  v12 = *(unsigned __int8 **)(v11 + 24);
  if ( v12 )
  {
LABEL_7:
    v13 = sub_3247C80((__int64)a1, v12);
    if ( v13 )
      sub_32494F0(a1, a2, 25, (unsigned __int64)v13);
    goto LABEL_9;
  }
LABEL_20:
  v25 = *(_QWORD *)(v11 + 32);
  if ( v25 )
  {
    v21 = sub_A777F0(0x10u, a1 + 11);
    v22 = v21;
    if ( v21 )
    {
      *(_QWORD *)v21 = 0;
      *(_DWORD *)(v21 + 8) = 0;
    }
    v23 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
    sub_3247620((__int64)v28, a1[23], v23, v22);
    v31 = v31 & 0xF8 | 2;
    v26 = *(unsigned __int64 **)(v25 + 16);
    v27 = *(_QWORD *)(v25 + 24);
    sub_3244870(v28, &v26);
    sub_3243D40((__int64)v28);
    sub_3249620(a1, a2, 25, v32);
    if ( v29 != v30 )
    {
      _libc_free((unsigned __int64)v29);
      v14 = *(_BYTE *)(a3 - 16);
      if ( (v14 & 2) != 0 )
        goto LABEL_10;
      goto LABEL_25;
    }
  }
  else
  {
    v24 = *(_QWORD *)(a3 + 24);
    BYTE2(v28[0]) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 11, v28[0], v24 >> 3);
  }
LABEL_9:
  v14 = *(_BYTE *)(a3 - 16);
  if ( (v14 & 2) != 0 )
  {
LABEL_10:
    v15 = *(_QWORD *)(a3 - 32);
    goto LABEL_11;
  }
LABEL_25:
  v15 = v3 - 8LL * ((v14 >> 2) & 0xF);
LABEL_11:
  v16 = *(_QWORD *)(v15 + 40);
  if ( v16 )
  {
    v17 = sub_A777F0(0x10u, a1 + 11);
    v18 = v17;
    if ( v17 )
    {
      *(_QWORD *)v17 = 0;
      *(_DWORD *)(v17 + 8) = 0;
    }
    v19 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
    sub_3247620((__int64)v28, a1[23], v19, v18);
    v31 = v31 & 0xF8 | 2;
    v26 = *(unsigned __int64 **)(v16 + 16);
    v27 = *(_QWORD *)(v16 + 24);
    sub_3244870(v28, &v26);
    sub_3243D40((__int64)v28);
    sub_3249620(a1, a2, 80, v32);
    if ( v29 != v30 )
      _libc_free((unsigned __int64)v29);
  }
  v20 = *(unsigned int *)(a3 + 44);
  if ( (_DWORD)v20 )
  {
    LODWORD(v28[0]) = 65547;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 62, 65547, v20);
  }
}
