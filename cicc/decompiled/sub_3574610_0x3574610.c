// Function: sub_3574610
// Address: 0x3574610
//
__int64 __fastcall sub_3574610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  __int64 v9; // rax
  char v10; // si
  int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 (*v15)(); // rdx
  unsigned __int64 *v16; // rdx
  __int64 v17; // rcx
  unsigned __int64 v18; // r12
  unsigned __int64 *v19; // rax
  char v20; // dl
  __int64 (*v21)(); // rax
  int v22; // eax
  char v23; // dl
  bool v24; // zf
  unsigned __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r14
  __int64 v28; // r15
  __int64 v29; // r12
  __int64 v30; // rax
  bool v31; // al
  int v32; // eax
  __int64 v33; // r15
  _BYTE *v34; // r12
  _BYTE *i; // r15
  __int64 v36; // r14
  __int64 v37; // rbx
  __int64 v38; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // r14
  _BYTE *v43; // rbx
  _BYTE *j; // r12
  unsigned __int8 v45; // [rsp+17h] [rbp-C9h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  _BYTE *v47; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE *v48; // [rsp+28h] [rbp-B8h]
  __int64 (__fastcall *v49)(_BYTE *); // [rsp+30h] [rbp-B0h]
  _BYTE *v50; // [rsp+38h] [rbp-A8h]
  _BYTE *v51; // [rsp+40h] [rbp-A0h]
  __int64 (__fastcall *v52)(_BYTE *); // [rsp+48h] [rbp-98h]
  __int64 v53; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 *v54; // [rsp+58h] [rbp-88h]
  __int64 v55; // [rsp+60h] [rbp-80h]
  int v56; // [rsp+68h] [rbp-78h]
  char v57; // [rsp+6Ch] [rbp-74h]
  char v58; // [rsp+70h] [rbp-70h] BYREF

  v8 = *(_DWORD *)(a1 + 1588);
  *(_DWORD *)(a1 + 1584) = 0;
  if ( v8 )
  {
    v9 = 0;
  }
  else
  {
    sub_C8D5F0(a1 + 1576, (const void *)(a1 + 1592), 1u, 8u, a5, a6);
    v9 = 8LL * *(unsigned int *)(a1 + 1584);
  }
  v10 = 1;
  v46 = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 1576) + v9) = a2;
  v11 = *(_DWORD *)(a1 + 1584);
  v54 = (unsigned __int64 *)&v58;
  v12 = *(_QWORD *)(a1 + 216);
  v13 = v11 + 1;
  v55 = 8;
  *(_DWORD *)(a1 + 1584) = v13;
  v56 = 0;
  v57 = 1;
  v14 = *(_QWORD *)(v12 + 16);
  v53 = 0;
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 128LL);
  if ( v15 != sub_2DAC790 )
  {
    v10 = v57;
    v46 = ((__int64 (__fastcall *)(__int64, __int64))v15)(v14, 1);
    v13 = *(_DWORD *)(a1 + 1584);
  }
  while ( v13 )
  {
    v16 = *(unsigned __int64 **)(a1 + 1576);
    v17 = v13;
    v18 = v16[v13 - 1];
    *(_DWORD *)(a1 + 1584) = v13 - 1;
    if ( !v10 )
      goto LABEL_15;
    v19 = v54;
    v17 = HIDWORD(v55);
    v16 = &v54[HIDWORD(v55)];
    if ( v54 != v16 )
    {
      while ( v18 != *v19 )
      {
        if ( v16 == ++v19 )
          goto LABEL_45;
      }
LABEL_11:
      v13 = *(_DWORD *)(a1 + 1584);
      continue;
    }
LABEL_45:
    if ( HIDWORD(v55) < (unsigned int)v55 )
    {
      ++HIDWORD(v55);
      *v16 = v18;
      ++v53;
    }
    else
    {
LABEL_15:
      sub_C8CC70((__int64)&v53, v18, (__int64)v16, v17, a5, a6);
      v10 = v57;
      if ( !v20 )
        goto LABEL_11;
    }
    v45 = sub_2E8B090(v18);
    if ( v45 )
      goto LABEL_66;
    v21 = *(__int64 (**)())(*(_QWORD *)v46 + 1536LL);
    if ( v21 != sub_2FDC850 && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v21)(v46, v18) )
      goto LABEL_57;
    v22 = *(_DWORD *)(v18 + 44);
    v23 = v22;
    if ( (v22 & 0xC) != 0 )
    {
      v24 = (v22 & 8) == 0;
      v25 = v18;
      if ( !v24 )
      {
        do
          v25 = *(_QWORD *)(v25 + 8);
        while ( (*(_BYTE *)(v25 + 44) & 8) != 0 );
      }
      v26 = *(_QWORD *)(v25 + 8);
      if ( (v23 & 4) != 0 )
      {
        do
          v18 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v18 + 44) & 4) != 0 );
      }
      v27 = *(_QWORD *)(v18 + 8);
      if ( v27 != v26 )
      {
        while ( 1 )
        {
          if ( *(_WORD *)(v27 + 68) == 68 || !*(_WORD *)(v27 + 68) )
          {
LABEL_30:
            v31 = v26 != v27;
            goto LABEL_31;
          }
          v28 = *(_QWORD *)(v27 + 32);
          v29 = v28 + 40LL * (*(_DWORD *)(v27 + 40) & 0xFFFFFF);
          v30 = v28 + 40LL * (unsigned int)sub_2E88F80(v27);
          if ( v29 != v30 )
          {
            while ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
            {
              v30 += 40;
              if ( v29 == v30 )
                goto LABEL_33;
            }
            goto LABEL_30;
          }
LABEL_33:
          if ( (unsigned int)*(unsigned __int16 *)(v27 + 68) - 1 > 1
            || (*(_BYTE *)(*(_QWORD *)(v27 + 32) + 64LL) & 8) == 0 )
          {
            v32 = *(_DWORD *)(v27 + 44);
            if ( (v32 & 4) != 0 || (v32 & 8) == 0 )
            {
              if ( (*(_QWORD *)(*(_QWORD *)(v27 + 16) + 24LL) & 0x80000LL) == 0 )
                goto LABEL_38;
            }
            else if ( !sub_2E88A90(v27, 0x80000, 1) )
            {
              goto LABEL_38;
            }
          }
          if ( !(unsigned __int8)sub_2E8AED0(v27) && *(_BYTE *)(a1 + 1640) )
            goto LABEL_30;
LABEL_38:
          v33 = *(_QWORD *)(v27 + 32);
          v34 = (_BYTE *)(v33 + 40LL * (*(_DWORD *)(v27 + 40) & 0xFFFFFF));
          for ( i = (_BYTE *)(v33 + 40LL * (unsigned int)sub_2E88FE0(v27)); v34 != i; i += 40 )
          {
            if ( (unsigned __int8)sub_2E2FA70(i) )
              break;
          }
          v47 = i;
          v48 = v34;
          v49 = sub_2E2FA70;
          v50 = v34;
          v51 = v34;
          v52 = sub_2E2FA70;
          v31 = sub_3574560((__int64 *)&v47, a1);
          if ( v31 )
            goto LABEL_30;
          v27 = *(_QWORD *)(v27 + 8);
          if ( v26 == v27 )
            goto LABEL_31;
        }
      }
      v10 = v57;
      v13 = *(_DWORD *)(a1 + 1584);
    }
    else
    {
      if ( !*(_WORD *)(v18 + 68) || *(_WORD *)(v18 + 68) == 68 )
        goto LABEL_57;
      v36 = *(_QWORD *)(v18 + 32);
      v37 = v36 + 40LL * (*(_DWORD *)(v18 + 40) & 0xFFFFFF);
      v38 = v36 + 40LL * (unsigned int)sub_2E88F80(v18);
      if ( v37 != v38 )
      {
        while ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
        {
          v38 += 40;
          if ( v38 == v37 )
            goto LABEL_62;
        }
LABEL_57:
        v10 = v57;
        goto LABEL_58;
      }
LABEL_62:
      if ( (unsigned int)*(unsigned __int16 *)(v18 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v18 + 32) + 64LL) & 8) != 0
        || ((v40 = *(_DWORD *)(v18 + 44), (v40 & 4) != 0) || (v40 & 8) == 0
          ? (v41 = (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL) >> 19) & 1LL)
          : (LOBYTE(v41) = sub_2E88A90(v18, 0x80000, 1)),
            (_BYTE)v41) )
      {
        if ( !(unsigned __int8)sub_2E8AED0(v18) && *(_BYTE *)(a1 + 1640) )
        {
LABEL_66:
          v45 = 0;
          v10 = v57;
LABEL_58:
          if ( v10 )
            return v45;
          goto LABEL_14;
        }
      }
      v42 = *(_QWORD *)(v18 + 32);
      v43 = (_BYTE *)(v42 + 40LL * (*(_DWORD *)(v18 + 40) & 0xFFFFFF));
      for ( j = (_BYTE *)(v42 + 40LL * (unsigned int)sub_2E88FE0(v18)); v43 != j; j += 40 )
      {
        if ( (unsigned __int8)sub_2E2FA70(j) )
          break;
      }
      v47 = j;
      v48 = v43;
      v49 = sub_2E2FA70;
      v50 = v43;
      v51 = v43;
      v52 = sub_2E2FA70;
      v31 = sub_3574560((__int64 *)&v47, a1);
LABEL_31:
      v10 = v57;
      if ( v31 )
        goto LABEL_58;
      v13 = *(_DWORD *)(a1 + 1584);
    }
  }
  v45 = 1;
  if ( !v10 )
LABEL_14:
    _libc_free((unsigned __int64)v54);
  return v45;
}
