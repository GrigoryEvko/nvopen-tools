// Function: sub_184F780
// Address: 0x184f780
//
__int64 __fastcall sub_184F780(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  _QWORD *v3; // r12
  char v5; // al
  __int64 *v6; // rbx
  __int64 *j; // r12
  __int64 v8; // r14
  unsigned __int64 *v9; // rax
  _QWORD *i; // r13
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  _QWORD *v13; // r8
  _QWORD *v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // ebx
  __int64 v19; // r14
  __int64 v20; // rcx
  _QWORD *v21; // r8
  _QWORD *v22; // r9
  char v23; // r13
  __int64 v24; // rdx
  int v25; // ecx
  __int64 v26; // rdi
  __int64 *v27; // r10
  unsigned __int64 v28; // r10
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  int v32; // ecx
  int v33; // r8d
  unsigned int v34; // eax
  __int64 v35; // rsi
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  _QWORD *v42; // r8
  _QWORD *v43; // r9
  __int64 v45; // [rsp+18h] [rbp-108h]
  _QWORD *v46; // [rsp+20h] [rbp-100h]
  __int64 v47; // [rsp+28h] [rbp-F8h]
  char v48; // [rsp+35h] [rbp-EBh]
  char v49; // [rsp+36h] [rbp-EAh]
  unsigned __int8 v50; // [rsp+37h] [rbp-E9h]
  __int64 v51; // [rsp+38h] [rbp-E8h]
  __int64 v52; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v53; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+58h] [rbp-C8h]
  __int64 v55; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE *v56; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v57; // [rsp+A8h] [rbp-78h]
  _BYTE v58[112]; // [rsp+B0h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 80);
  v51 = v1 + 8LL * *(unsigned int *)(a1 + 88);
  if ( v1 == v51 )
    return 0;
  v2 = *(_QWORD *)(a1 + 80);
  v50 = 0;
  v48 = 1;
LABEL_3:
  v3 = *(_QWORD **)v2;
  v53 = *(_QWORD *)(*(_QWORD *)v2 + 112LL);
  if ( (unsigned __int8)sub_1560260(&v53, 0, 32) )
    goto LABEL_9;
  if ( sub_15E4F60((__int64)v3) )
    return 0;
  sub_15E4B50((__int64)v3);
  v49 = v5;
  if ( v5 )
    return 0;
  if ( *(_BYTE *)(**(_QWORD **)(v3[3] + 16LL) + 8LL) != 15 )
    goto LABEL_9;
  v9 = (unsigned __int64 *)&v55;
  v53 = 0;
  v54 = 1;
  do
    *v9++ = -8;
  while ( v9 != (unsigned __int64 *)&v56 );
  v56 = v58;
  v57 = 0x800000000LL;
  for ( i = (_QWORD *)v3[10]; v3 + 9 != i; i = (_QWORD *)i[1] )
  {
    v11 = (__int64)(i - 3);
    if ( !i )
      v11 = 0;
    v12 = sub_157EBA0(v11);
    if ( *(_BYTE *)(v12 + 16) == 25 )
    {
      v15 = 0;
      v16 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
      if ( (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) != 0 )
      {
        v16 = -3LL * (unsigned int)v16;
        v15 = *(_QWORD *)(v12 + 8 * v16);
      }
      v52 = v15;
      sub_184F4A0((__int64)&v53, &v52, v16, v15, v13, v14);
    }
  }
  v47 = sub_1632FA0(v3[5]);
  v17 = 0;
  if ( !(_DWORD)v57 )
    goto LABEL_46;
  v46 = v3;
  v18 = 0;
  v45 = v2;
  while ( 2 )
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)&v56[8 * v17];
      v23 = sub_14BFF20(v19, v47, 0, 0, 0, 0);
      if ( !v23 )
        break;
LABEL_44:
      v17 = (unsigned int)(v18 + 1);
      v18 = v17;
      if ( (_DWORD)v57 == (_DWORD)v17 )
        goto LABEL_45;
    }
    v24 = *(unsigned __int8 *)(v19 + 16);
    if ( (unsigned __int8)v24 <= 0x17u )
      goto LABEL_33;
    switch ( (char)v24 )
    {
      case 29:
      case 78:
        v28 = v19 | 4;
        if ( (_BYTE)v24 == 78 )
          goto LABEL_70;
        v29 = 0;
        if ( (_BYTE)v24 != 29 )
          goto LABEL_52;
        v28 = v19 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_70:
        v29 = v28 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v28 & 4) != 0 )
          v30 = v29 - 24;
        else
LABEL_52:
          v30 = v29 - 72;
        v31 = *(_QWORD *)v30;
        if ( *(_BYTE *)(*(_QWORD *)v30 + 16LL) )
          goto LABEL_33;
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v32 = 7;
          v26 = a1 + 16;
        }
        else
        {
          v25 = *(_DWORD *)(a1 + 24);
          v26 = *(_QWORD *)(a1 + 16);
          if ( !v25 )
            goto LABEL_33;
          v32 = v25 - 1;
        }
        v33 = 1;
        v34 = v32 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v35 = *(_QWORD *)(v26 + 8LL * v34);
        if ( v31 == v35 )
        {
LABEL_57:
          v17 = (unsigned int)(v18 + 1);
          v49 = 1;
          v18 = v17;
          if ( (_DWORD)v57 == (_DWORD)v17 )
          {
LABEL_45:
            v3 = v46;
            v2 = v45;
LABEL_46:
            v23 = 1;
            goto LABEL_34;
          }
          continue;
        }
        while ( v35 != -8 )
        {
          v34 = v32 & (v33 + v34);
          v35 = *(_QWORD *)(v26 + 8LL * v34);
          if ( v31 == v35 )
            goto LABEL_57;
          ++v33;
        }
LABEL_33:
        v3 = v46;
        v2 = v45;
LABEL_34:
        if ( v56 != v58 )
          _libc_free((unsigned __int64)v56);
        if ( (v54 & 1) == 0 )
          j___libc_free_0(v55);
        if ( !v23 )
        {
          v48 = 0;
          v2 += 8;
          if ( v51 == v2 )
            goto LABEL_10;
          goto LABEL_3;
        }
        if ( v49 )
        {
LABEL_9:
          v2 += 8;
          if ( v51 == v2 )
            goto LABEL_10;
          goto LABEL_3;
        }
        v2 += 8;
        sub_15E0D50((__int64)v3, 0, 32);
        v50 = v23;
        if ( v51 != v2 )
          goto LABEL_3;
LABEL_10:
        if ( v48 )
        {
          v6 = *(__int64 **)(a1 + 80);
          for ( j = &v6[*(unsigned int *)(a1 + 88)]; j != v6; ++v6 )
          {
            v8 = *v6;
            v53 = *(_QWORD *)(*v6 + 112);
            if ( !(unsigned __int8)sub_1560260(&v53, 0, 32)
              && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v8 + 24) + 16LL) + 8LL) == 15 )
            {
              sub_15E0D50(v8, 0, 32);
              v50 = v48;
            }
          }
        }
        return v50;
      case 56:
      case 71:
      case 72:
        if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
          v27 = *(__int64 **)(v19 - 8);
        else
          v27 = (__int64 *)(v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF));
        v52 = *v27;
        sub_184F4A0((__int64)&v53, &v52, v24, v20, v21, v22);
        goto LABEL_44;
      case 77:
        if ( (*(_DWORD *)(v19 + 20) & 0xFFFFFFF) == 0 )
          goto LABEL_44;
        v36 = 0;
        v37 = 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
        do
        {
          if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
          {
            v38 = *(_QWORD *)(v19 - 8);
          }
          else
          {
            v20 = 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
            v38 = v19 - v20;
          }
          v39 = *(_QWORD *)(v38 + v36);
          v36 += 24;
          v52 = v39;
          sub_184F4A0((__int64)&v53, &v52, v39, v20, v21, v22);
        }
        while ( v37 != v36 );
        v17 = (unsigned int)(v18 + 1);
        v18 = v17;
        if ( (_DWORD)v57 == (_DWORD)v17 )
          goto LABEL_45;
        continue;
      case 79:
        v52 = *(_QWORD *)(v19 - 48);
        sub_184F4A0((__int64)&v53, &v52, v24, v20, v21, v22);
        v52 = *(_QWORD *)(v19 - 24);
        sub_184F4A0((__int64)&v53, &v52, v40, v41, v42, v43);
        v17 = (unsigned int)(v18 + 1);
        v18 = v17;
        if ( (_DWORD)v57 == (_DWORD)v17 )
          goto LABEL_45;
        continue;
      default:
        goto LABEL_33;
    }
  }
}
