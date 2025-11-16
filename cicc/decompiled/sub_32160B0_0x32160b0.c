// Function: sub_32160B0
// Address: 0x32160b0
//
__int64 __fastcall sub_32160B0(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  unsigned __int64 v3; // r15
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  const char *v9; // rax
  __int64 v10; // rdx
  const char *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // r13
  const char *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // r15
  __int64 v18; // r9
  __int64 *v19; // rax
  _BYTE *v20; // rsi
  __int64 *v21; // rdi
  const char *i; // rax
  __int64 v23; // rdi
  void (*v24)(); // r10
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+8h] [rbp-98h]
  void (*v28)(); // [rsp+10h] [rbp-90h]
  void (*v29)(); // [rsp+10h] [rbp-90h]
  void (*v30)(); // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v32[9]; // [rsp+37h] [rbp-69h] BYREF
  const char *v33; // [rsp+40h] [rbp-60h] BYREF
  __int64 v34; // [rsp+48h] [rbp-58h]
  const char *v35; // [rsp+50h] [rbp-50h]
  __int16 v36; // [rsp+60h] [rbp-40h]

  v3 = a2;
  result = (__int64)&v31;
  if ( (__int64 *)a2 != a3 )
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v3 + 8) != 1 )
        goto LABEL_7;
      v7 = *(_QWORD *)(v3 + 16);
      if ( (unsigned int)v7 <= 0x23 )
        break;
      if ( (v7 & 0xFFFFFFFD) != 0x90 )
        goto LABEL_7;
      v27 = *(_QWORD *)(a1 + 224);
      v30 = *(void (**)())(*(_QWORD *)v27 + 120LL);
      v14 = sub_E06E20(v7);
      v36 = 261;
      v33 = v14;
      v34 = v15;
      if ( v30 != nullsub_98 )
        ((void (__fastcall *)(__int64, const char **, __int64))v30)(v27, &v33, 1);
      sub_3215FD0(v3 + 8, a1);
      v16 = *(_QWORD *)v3;
      v17 = 0;
      if ( (v16 & 4) == 0 )
        v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (__int64 *)v17 == a3 || (v18 = v17 + 8, *(_WORD *)(v17 + 14) != 15) || *(_DWORD *)(v17 + 8) != 1 )
LABEL_40:
        BUG();
      if ( *(_QWORD *)(v17 + 16) )
      {
        v31 = *(_QWORD *)(v17 + 16);
        v19 = (__int64 *)((char *)&v31 + 7);
        v20 = v32;
        do
        {
          *v20++ = *(_BYTE *)v19;
          v21 = v19;
          v19 = (__int64 *)((char *)v19 - 1);
        }
        while ( v21 != &v31 );
        v32[8] = 0;
        for ( i = v32; !*i; ++i )
          ;
        v23 = *(_QWORD *)(a1 + 224);
        v24 = *(void (**)())(*(_QWORD *)v23 + 120LL);
        v33 = i;
        v35 = " [unsigned LEB]";
        v36 = 771;
        if ( v24 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v24)(v23, &v33, 1);
          v18 = v17 + 8;
        }
      }
      sub_3215FD0(v18, a1);
      result = *(_QWORD *)v17;
      v3 = 0;
      if ( (result & 4) == 0 )
        v3 = result & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_DWORD)v7 == 146 )
      {
        if ( (__int64 *)v3 == a3 )
          goto LABEL_40;
        result = sub_3215FD0(v3 + 8, a1);
        v8 = *(_QWORD *)v3;
        if ( (*(_QWORD *)v3 & 4) != 0 )
        {
          v3 = 0;
          goto LABEL_9;
        }
        goto LABEL_8;
      }
LABEL_9:
      if ( (__int64 *)v3 == a3 )
        return result;
    }
    if ( (unsigned int)v7 > 2 )
    {
      switch ( (int)v7 )
      {
        case 3:
        case 16:
        case 35:
          v26 = *(_QWORD *)(a1 + 224);
          v29 = *(void (**)())(*(_QWORD *)v26 + 120LL);
          v11 = sub_E06E20(v7);
          v36 = 261;
          v33 = v11;
          v34 = v12;
          if ( v29 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v29)(v26, &v33, 1);
          v13 = 0;
          sub_3215FD0(v3 + 8, a1);
          if ( (*(_QWORD *)v3 & 4) == 0 )
            v13 = (__int64 *)(*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v13 == a3 )
            goto LABEL_40;
          result = sub_3215FD0((__int64)(v13 + 1), a1);
          v8 = *v13;
          if ( (*v13 & 4) != 0 )
            goto LABEL_19;
          goto LABEL_8;
        case 6:
        case 18:
        case 24:
        case 34:
          v25 = *(_QWORD *)(a1 + 224);
          v28 = *(void (**)())(*(_QWORD *)v25 + 120LL);
          v9 = sub_E06E20(v7);
          v36 = 261;
          v33 = v9;
          v34 = v10;
          if ( v28 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v28)(v25, &v33, 1);
          break;
        default:
          break;
      }
    }
LABEL_7:
    result = sub_3215FD0(v3 + 8, a1);
    v8 = *(_QWORD *)v3;
    if ( (*(_QWORD *)v3 & 4) != 0 )
    {
LABEL_19:
      v3 = 0;
      goto LABEL_9;
    }
LABEL_8:
    v3 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_9;
  }
  return result;
}
