// Function: sub_184FD30
// Address: 0x184fd30
//
__int64 __fastcall sub_184FD30(__int64 a1)
{
  __int64 *v1; // r12
  __int64 v2; // rbx
  __int64 *v3; // rsi
  unsigned int v4; // r14d
  char v6; // al
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // r12
  __int64 *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  _QWORD *v17; // r8
  _QWORD *v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rdx
  _BYTE *v21; // rdi
  __int64 v22; // rax
  int v23; // r15d
  __int64 v24; // r13
  unsigned __int8 v25; // al
  bool v26; // al
  char v27; // bl
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // rbx
  __int64 v33; // rdx
  int v34; // ecx
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rsi
  char v38; // al
  __int64 *v39; // r13
  __int64 v40; // rdx
  __int64 v41; // rcx
  _QWORD *v42; // r9
  __int64 v43; // rax
  __int64 *v44; // rbx
  __int64 *v45; // r8
  __int64 *v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rax
  int v49; // ecx
  int v50; // r8d
  __int64 *v51; // [rsp+8h] [rbp-108h]
  __int64 *v53; // [rsp+28h] [rbp-E8h]
  __int64 v54; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v55; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v56; // [rsp+48h] [rbp-C8h]
  __int64 v57; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE *v58; // [rsp+90h] [rbp-80h] BYREF
  __int64 v59; // [rsp+98h] [rbp-78h]
  _BYTE v60[112]; // [rsp+A0h] [rbp-70h] BYREF

  v1 = *(__int64 **)(a1 + 80);
  v53 = &v1[*(unsigned int *)(a1 + 88)];
  if ( v1 == v53 )
    return 0;
  while ( 2 )
  {
    while ( 1 )
    {
      v2 = *v1;
      v3 = 0;
      if ( !(unsigned __int8)sub_1560260((_QWORD *)(*v1 + 112), 0, 20) )
      {
        if ( sub_15E4F60(v2) )
          return 0;
        sub_15E4B50(v2);
        if ( v6 )
          return 0;
        if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v2 + 24) + 16LL) + 8LL) == 15 )
          break;
      }
LABEL_8:
      if ( v53 == ++v1 )
        goto LABEL_9;
    }
    v55 = 0;
    v10 = &v57;
    v56 = 1;
    do
      *v10++ = -8;
    while ( v10 != (__int64 *)&v58 );
    v11 = v2 + 72;
    v58 = v60;
    v59 = 0x800000000LL;
    v12 = *(_QWORD *)(v11 + 8);
    if ( v12 == v11 )
    {
      if ( (v56 & 1) == 0 )
      {
        v27 = 1;
        goto LABEL_34;
      }
      goto LABEL_8;
    }
    do
    {
      v13 = v12 - 24;
      if ( !v12 )
        v13 = 0;
      v14 = sub_157EBA0(v13);
      if ( *(_BYTE *)(v14 + 16) == 25 )
      {
        v19 = 0;
        v20 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
        if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) != 0 )
        {
          v20 = -3LL * (unsigned int)v20;
          v19 = *(_QWORD *)(v14 + 8 * v20);
        }
        v3 = &v54;
        v54 = v19;
        sub_184F4A0((__int64)&v55, &v54, v20, v19, v17, v18);
      }
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v11 != v12 );
    v21 = v58;
    if ( !(_DWORD)v59 )
    {
LABEL_51:
      v27 = 1;
      goto LABEL_31;
    }
    v22 = 0;
    v23 = 0;
LABEL_28:
    while ( 1 )
    {
      v24 = *(_QWORD *)&v21[8 * v22];
      v25 = *(_BYTE *)(v24 + 16);
      if ( v25 > 0x10u )
        break;
      v26 = sub_1593BB0(v24, (__int64)v3, v15, v16);
      v21 = v58;
      v27 = v26;
      if ( !v26 && *(_BYTE *)(v24 + 16) != 9 )
        goto LABEL_31;
LABEL_50:
      v22 = (unsigned int)(v23 + 1);
      v23 = v22;
      if ( (_DWORD)v59 == (_DWORD)v22 )
        goto LABEL_51;
    }
    if ( v25 == 17 )
    {
LABEL_55:
      v27 = 0;
      goto LABEL_31;
    }
    if ( v25 <= 0x17u )
    {
LABEL_49:
      v3 = 0;
      v38 = sub_139D0F0(v24, 0);
      v21 = v58;
      if ( v38 )
        goto LABEL_55;
      goto LABEL_50;
    }
    v15 = (unsigned int)v25 - 29;
    switch ( v25 )
    {
      case 0x1Du:
      case 0x4Eu:
        if ( v25 == 78 )
        {
          v28 = v24 | 4;
        }
        else
        {
          v29 = 0;
          v30 = 56;
          if ( v25 != 29 )
            goto LABEL_72;
          v28 = v24 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v29 = v28 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = (v28 & 0xFFFFFFFFFFFFFFF8LL) + 56;
        if ( (v28 & 4) != 0 )
        {
          if ( (unsigned __int8)sub_1560260((_QWORD *)v30, 0, 20) )
            goto LABEL_49;
          v31 = *(_QWORD *)(v29 - 24);
          if ( !*(_BYTE *)(v31 + 16) )
          {
            v54 = *(_QWORD *)(v31 + 112);
            if ( (unsigned __int8)sub_1560260(&v54, 0, 20) )
              goto LABEL_49;
          }
          v32 = v29 - 24;
          goto LABEL_45;
        }
LABEL_72:
        if ( (unsigned __int8)sub_1560260((_QWORD *)v30, 0, 20) )
          goto LABEL_49;
        v48 = *(_QWORD *)(v29 - 72);
        if ( !*(_BYTE *)(v48 + 16) )
        {
          v54 = *(_QWORD *)(v48 + 112);
          if ( (unsigned __int8)sub_1560260(&v54, 0, 20) )
            goto LABEL_49;
        }
        v32 = v29 - 72;
LABEL_45:
        v33 = *(_QWORD *)v32;
        if ( *(_BYTE *)(*(_QWORD *)v32 + 16LL) )
          goto LABEL_80;
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v34 = 7;
          v35 = a1 + 16;
        }
        else
        {
          v49 = *(_DWORD *)(a1 + 24);
          v35 = *(_QWORD *)(a1 + 16);
          if ( !v49 )
            goto LABEL_80;
          v34 = v49 - 1;
        }
        v36 = v34 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v37 = *(_QWORD *)(v35 + 8LL * v36);
        if ( v37 == v33 )
          goto LABEL_49;
        v50 = 1;
        while ( v37 != -8 )
        {
          v36 = v34 & (v50 + v36);
          v37 = *(_QWORD *)(v35 + 8LL * v36);
          if ( v37 == v33 )
            goto LABEL_49;
          ++v50;
        }
LABEL_80:
        v21 = v58;
        v27 = 0;
LABEL_31:
        if ( v21 != v60 )
          _libc_free((unsigned __int64)v21);
        if ( (v56 & 1) == 0 )
LABEL_34:
          j___libc_free_0(v57);
        if ( !v27 )
          return 0;
        if ( v53 != ++v1 )
          continue;
LABEL_9:
        v7 = *(__int64 **)(a1 + 80);
        v8 = &v7[*(unsigned int *)(a1 + 88)];
        if ( v8 == v7 )
          return 0;
        v4 = 0;
        do
        {
          v9 = *v7;
          if ( !(unsigned __int8)sub_1560260((_QWORD *)(*v7 + 112), 0, 20)
            && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v9 + 24) + 16LL) + 8LL) == 15 )
          {
            v4 = 1;
            sub_15E0D50(v9, 0, 20);
          }
          ++v7;
        }
        while ( v8 != v7 );
        return v4;
      case 0x35u:
        goto LABEL_49;
      case 0x38u:
      case 0x47u:
      case 0x48u:
        if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
          v39 = *(__int64 **)(v24 - 8);
        else
          v39 = (__int64 *)(v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF));
        v3 = &v54;
        v54 = *v39;
        sub_184F4A0((__int64)&v55, &v54, v15, v16, &v54, v18);
        v22 = (unsigned int)(v23 + 1);
        v21 = v58;
        v23 = v22;
        if ( (_DWORD)v59 == (_DWORD)v22 )
          goto LABEL_51;
        goto LABEL_28;
      case 0x4Du:
        v43 = 3LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
        {
          v44 = *(__int64 **)(v24 - 8);
          v24 = (__int64)&v44[v43];
        }
        else
        {
          v44 = (__int64 *)(v24 - v43 * 8);
        }
        if ( v44 == (__int64 *)v24 )
          goto LABEL_50;
        v45 = &v54;
        v51 = v1;
        v46 = v44;
        do
        {
          v47 = *v46;
          v3 = &v54;
          v46 += 3;
          v54 = v47;
          sub_184F4A0((__int64)&v55, &v54, v47, v16, v45, v18);
        }
        while ( (__int64 *)v24 != v46 );
        v1 = v51;
        v21 = v58;
        v22 = (unsigned int)(v23 + 1);
        v23 = v22;
        if ( (_DWORD)v59 == (_DWORD)v22 )
          goto LABEL_51;
        goto LABEL_28;
      case 0x4Fu:
        v54 = *(_QWORD *)(v24 - 48);
        sub_184F4A0((__int64)&v55, &v54, v15, v16, &v54, v18);
        v3 = &v54;
        v54 = *(_QWORD *)(v24 - 24);
        sub_184F4A0((__int64)&v55, v3, v40, v41, v3, v42);
        v22 = (unsigned int)(v23 + 1);
        v21 = v58;
        v23 = v22;
        if ( (_DWORD)v59 == (_DWORD)v22 )
          goto LABEL_51;
        goto LABEL_28;
      default:
        goto LABEL_55;
    }
  }
}
