// Function: sub_2589AC0
// Address: 0x2589ac0
//
_BOOL8 __fastcall sub_2589AC0(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 *v5; // r10
  unsigned int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 v12; // r13
  _BOOL4 v13; // r12d
  char v15; // [rsp+1Bh] [rbp-155h] BYREF
  int v16; // [rsp+1Ch] [rbp-154h] BYREF
  _QWORD v17[4]; // [rsp+20h] [rbp-150h] BYREF
  void *v18; // [rsp+40h] [rbp-130h] BYREF
  int v19; // [rsp+48h] [rbp-128h]
  const void *v20; // [rsp+50h] [rbp-120h] BYREF
  unsigned int v21; // [rsp+58h] [rbp-118h]
  const void *v22; // [rsp+60h] [rbp-110h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-108h]
  const void *v24; // [rsp+70h] [rbp-100h] BYREF
  unsigned int v25; // [rsp+78h] [rbp-F8h]
  const void *v26; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v27; // [rsp+88h] [rbp-E8h]
  void *v28; // [rsp+90h] [rbp-E0h] BYREF
  int v29; // [rsp+98h] [rbp-D8h]
  const void *v30; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v31; // [rsp+A8h] [rbp-C8h]
  const void *v32; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int v33; // [rsp+B8h] [rbp-B8h]
  const void *v34; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned int v35; // [rsp+C8h] [rbp-A8h]
  const void *v36; // [rsp+D0h] [rbp-A0h] BYREF
  unsigned int v37; // [rsp+D8h] [rbp-98h]
  const void *v38[18]; // [rsp+E0h] [rbp-90h] BYREF

  sub_AADB10((__int64)v38, *(_DWORD *)(a1 + 96), 0);
  v18 = &unk_4A16D38;
  v19 = (int)v38[1];
  v21 = (unsigned int)v38[1];
  if ( LODWORD(v38[1]) > 0x40 )
  {
    sub_C43780((__int64)&v20, v38);
    v23 = (unsigned int)v38[3];
    if ( LODWORD(v38[3]) <= 0x40 )
      goto LABEL_3;
LABEL_36:
    sub_C43780((__int64)&v22, &v38[2]);
    goto LABEL_4;
  }
  v20 = v38[0];
  v23 = (unsigned int)v38[3];
  if ( LODWORD(v38[3]) > 0x40 )
    goto LABEL_36;
LABEL_3:
  v22 = v38[2];
LABEL_4:
  sub_AADB10((__int64)&v24, (unsigned int)v38[1], 1);
  if ( LODWORD(v38[3]) > 0x40 && v38[2] )
    j_j___libc_free_0_0((unsigned __int64)v38[2]);
  if ( LODWORD(v38[1]) > 0x40 && v38[0] )
    j_j___libc_free_0_0((unsigned __int64)v38[0]);
  v4 = *(_QWORD *)(a1 + 80);
  v5 = (__int64 *)(a1 + 72);
  if ( v4 )
  {
    v6 = sub_250CB50((__int64 *)(a1 + 72), 0);
    v7 = sub_254C9B0(v4, v6);
    v9 = sub_2589400(a2, v7, v8, a1, 0, 0, 1);
    v5 = (__int64 *)(a1 + 72);
    v10 = v9;
    if ( v9 )
    {
      v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL);
      if ( v11 == sub_2534AC0 )
      {
        v12 = v10 + 88;
        LODWORD(v38[1]) = *(_DWORD *)(v10 + 112);
        if ( LODWORD(v38[1]) <= 0x40 )
          goto LABEL_14;
      }
      else
      {
        v12 = ((__int64 (__fastcall *)(__int64, __int64))v11)(v10, v7);
        LODWORD(v38[1]) = *(_DWORD *)(v12 + 24);
        if ( LODWORD(v38[1]) <= 0x40 )
        {
LABEL_14:
          v38[0] = *(const void **)(v12 + 16);
          LODWORD(v38[3]) = *(_DWORD *)(v12 + 40);
          if ( LODWORD(v38[3]) <= 0x40 )
          {
LABEL_15:
            v38[2] = *(const void **)(v12 + 32);
LABEL_16:
            sub_254F7F0((__int64)&v18, (__int64)v38);
            sub_969240((__int64 *)&v38[2]);
            sub_969240((__int64 *)v38);
            v38[0] = &unk_4A16D38;
            LODWORD(v38[1]) = v19;
            LODWORD(v38[3]) = v21;
            if ( v21 > 0x40 )
              sub_C43780((__int64)&v38[2], &v20);
            else
              v38[2] = v20;
            LODWORD(v38[5]) = v23;
            if ( v23 > 0x40 )
              sub_C43780((__int64)&v38[4], &v22);
            else
              v38[4] = v22;
            LODWORD(v38[7]) = v25;
            if ( v25 > 0x40 )
              sub_C43780((__int64)&v38[6], &v24);
            else
              v38[6] = v24;
            LODWORD(v38[9]) = v27;
            if ( v27 > 0x40 )
              sub_C43780((__int64)&v38[8], &v26);
            else
              v38[8] = v26;
            goto LABEL_24;
          }
LABEL_53:
          sub_C43780((__int64)&v38[2], (const void **)(v12 + 32));
          goto LABEL_16;
        }
      }
      sub_C43780((__int64)v38, (const void **)(v12 + 16));
      LODWORD(v38[3]) = *(_DWORD *)(v12 + 40);
      if ( LODWORD(v38[3]) <= 0x40 )
        goto LABEL_15;
      goto LABEL_53;
    }
  }
  memset(v38, 0, 0x58u);
  v16 = sub_250CB50(v5, 0);
  v17[0] = &v16;
  v17[1] = a2;
  v17[2] = a1;
  v17[3] = v38;
  v15 = 0;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_258A330,
                          (__int64)v17,
                          a1,
                          1u,
                          &v15) )
  {
    if ( !LOBYTE(v38[10]) )
      goto LABEL_25;
    v29 = (int)v38[3];
    if ( LODWORD(v38[3]) > 0x40 )
      sub_C43780((__int64)&v28, &v38[2]);
    else
      v28 = (void *)v38[2];
    v31 = (unsigned int)v38[5];
    if ( LODWORD(v38[5]) > 0x40 )
      sub_C43780((__int64)&v30, &v38[4]);
    else
      v30 = v38[4];
    sub_254F7F0((__int64)&v18, (__int64)&v28);
    sub_969240((__int64 *)&v30);
    sub_969240((__int64 *)&v28);
    v28 = &unk_4A16D38;
    v29 = v19;
    v31 = v21;
    if ( v21 > 0x40 )
      sub_C43780((__int64)&v30, &v20);
    else
      v30 = v20;
    v33 = v23;
    if ( v23 > 0x40 )
      sub_C43780((__int64)&v32, &v22);
    else
      v32 = v22;
    v35 = v25;
    if ( v25 > 0x40 )
      sub_C43780((__int64)&v34, &v24);
    else
      v34 = v24;
    v37 = v27;
    if ( v27 > 0x40 )
      sub_C43780((__int64)&v36, &v26);
    else
      v36 = v26;
    sub_253FFA0((__int64)&v28);
  }
  else
  {
    if ( v21 <= 0x40 && v25 <= 0x40 )
    {
      v21 = v25;
      v20 = v24;
    }
    else
    {
      sub_C43990((__int64)&v20, (__int64)&v24);
    }
    if ( v23 <= 0x40 && v27 <= 0x40 )
    {
      v23 = v27;
      v22 = v26;
    }
    else
    {
      sub_C43990((__int64)&v22, (__int64)&v26);
    }
  }
  if ( LOBYTE(v38[10]) )
  {
    LOBYTE(v38[10]) = 0;
LABEL_24:
    sub_253FFA0((__int64)v38);
  }
LABEL_25:
  v13 = sub_255B670(a1 + 88, (__int64)&v18);
  sub_253FFA0((__int64)&v18);
  return v13;
}
