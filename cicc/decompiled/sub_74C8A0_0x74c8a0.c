// Function: sub_74C8A0
// Address: 0x74c8a0
//
_QWORD *__fastcall sub_74C8A0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        int a5,
        _QWORD *a6,
        _DWORD *a7,
        __int64 *a8,
        _DWORD *a9,
        __int64 a10)
{
  __int64 v10; // rax
  char v11; // bl
  int v13; // r15d
  __int64 v14; // r12
  unsigned __int8 v15; // dl
  __int64 v16; // rdi
  __int64 v17; // r8
  int v18; // ebx
  int v19; // r10d
  void *v20; // r15
  __int64 v22; // rax
  __int64 m; // r12
  __int64 v24; // r15
  char v25; // al
  __int64 j; // rdi
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // r9
  __int64 k; // rax
  void *v34; // r15
  void (__fastcall *v35)(const char *); // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 i; // rbx
  __int64 v43; // rax
  char v44; // dl
  __int64 v45; // rax
  __int64 v46; // [rsp+0h] [rbp-80h]
  __int64 v47; // [rsp+8h] [rbp-78h]
  signed __int64 v48; // [rsp+8h] [rbp-78h]
  __int64 v50; // [rsp+18h] [rbp-68h]
  int v51; // [rsp+20h] [rbp-60h]
  __int64 v52; // [rsp+20h] [rbp-60h]
  void *v54; // [rsp+30h] [rbp-50h]
  _DWORD v56[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v10 = *(_QWORD *)(a1 + 192);
  *a9 = 0;
  *a7 = 0;
  *a8 = v10;
  v11 = *(_BYTE *)(a1 + 176);
  v50 = v10;
  v13 = a4;
  v14 = a2;
  switch ( v11 )
  {
    case 0:
      v11 = 7;
      v15 = 11;
      v17 = 0;
      v16 = *(_QWORD *)(a1 + 184);
      v54 = *(void **)(v16 + 152);
      *a7 = 1;
      goto LABEL_3;
    case 1:
      v16 = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(v16 + 172) & 2) != 0 && (!*(_BYTE *)(a10 + 136) || !*(_BYTE *)(a10 + 141)) )
      {
        for ( i = *(_QWORD *)(v16 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v43 = sub_745C70(i, a2, a3);
        v16 = v43;
        if ( v43 )
        {
          v54 = *(void **)(v43 + 120);
        }
        else
        {
          a4 = *(_QWORD *)(i + 160);
          if ( !a4 )
            BUG();
          v16 = *(_QWORD *)(i + 160);
          do
          {
            v44 = *(_BYTE *)(*(_QWORD *)(v16 + 120) + 140LL);
            if ( v44 == 12 )
            {
              v45 = *(_QWORD *)(v16 + 120);
              do
              {
                v45 = *(_QWORD *)(v45 + 160);
                v44 = *(_BYTE *)(v45 + 140);
              }
              while ( v44 == 12 );
            }
            if ( (unsigned __int8)(v44 - 9) > 1u )
            {
              v54 = *(void **)(v16 + 120);
              goto LABEL_82;
            }
            v16 = *(_QWORD *)(v16 + 112);
          }
          while ( v16 );
          v16 = *(_QWORD *)(i + 160);
          v54 = *(void **)(a4 + 120);
        }
LABEL_82:
        v11 = 7;
        v15 = 8;
        v17 = 0;
        goto LABEL_3;
      }
      v11 = 7;
      v15 = 7;
      v17 = 0;
      v54 = *(void **)(v16 + 120);
LABEL_3:
      if ( !a2 )
        v14 = (__int64)v54;
      if ( a5 )
      {
        if ( v16 )
        {
          sub_74C550(v16, v15, a10);
        }
        else if ( v17 )
        {
          sub_748000(v17, 0, a10, a4, v17);
        }
        else if ( v11 == 4 )
        {
          sub_74BEE0(a1, (void (__fastcall **)(const char *))a10);
        }
        else
        {
          if ( v11 != 5 )
LABEL_107:
            sub_721090();
          sub_74BF80(a1, (void (__fastcall **)(const char *))a10);
        }
      }
      v56[0] = 0;
      if ( (*(_BYTE *)(a1 + 168) & 8) == 0 || (v18 = sub_745A90((__int64)v54, v14, a3, v56)) != 0 )
      {
        v18 = 1;
        if ( !v50 )
        {
          v19 = v56[0];
          *a9 = 1;
          if ( v19 )
          {
            v20 = v54;
            *a7 = 1;
            goto LABEL_12;
          }
          goto LABEL_48;
        }
      }
      if ( v13 )
      {
        v34 = v54;
LABEL_46:
        if ( !*a9 )
          goto LABEL_13;
        v54 = v34;
LABEL_48:
        if ( !*a7 )
          goto LABEL_14;
        goto LABEL_49;
      }
      v51 = 0;
      v22 = v14;
      m = (__int64)v54;
      v24 = v22;
      break;
    case 2:
    case 3:
      v11 = 7;
      v15 = 0;
      v16 = 0;
      v17 = *(_QWORD *)(a1 + 184);
      v54 = *(void **)(v17 + 128);
      goto LABEL_3;
    case 4:
    case 5:
      if ( (*(_BYTE *)(a1 + 168) & 8) != 0 )
      {
        v15 = 0;
        v54 = &unk_4F07F40;
        v17 = 0;
        memset(&unk_4F07F40, 0, 0xC0u);
        a4 = 0;
        byte_4F07FCC = 21;
        v16 = 0;
      }
      else
      {
        v40 = sub_8D46C0(*(_QWORD *)(a1 + 128));
        v15 = 0;
        v16 = 0;
        v17 = 0;
        v54 = (void *)v40;
      }
      goto LABEL_3;
    case 6:
      v41 = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(a1 + 168) & 8) != 0 )
      {
        v54 = &unk_4F07F40;
        memset(&unk_4F07F40, 0, 0xC0u);
        a4 = 0;
        byte_4F07FCC = 21;
      }
      else
      {
        v54 = (void *)sub_8D46C0(*(_QWORD *)(a1 + 128));
      }
      v16 = v41;
      v15 = 12;
      v11 = 7;
      v17 = 0;
      goto LABEL_3;
    default:
      goto LABEL_107;
  }
  while ( 1 )
  {
    v25 = *(_BYTE *)(m + 140);
    for ( j = m; v25 == 12; v25 = *(_BYTE *)(j + 140) )
      j = *(_QWORD *)(j + 160);
    if ( v25 != 8 )
    {
      if ( (unsigned __int8)(v25 - 9) <= 1u )
      {
        v28 = *a8;
        if ( *a8 < 0 || v28 >= *(_QWORD *)(j + 128) || (v29 = *(_QWORD *)(j + 160)) == 0 )
        {
LABEL_45:
          v34 = (void *)m;
          goto LABEL_46;
        }
        while ( 1 )
        {
          v30 = *(_QWORD *)(v29 + 128);
          v31 = v30;
          if ( v28 >= v30 )
          {
            v32 = *(_QWORD *)(v29 + 120);
            for ( k = v32; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
              ;
            if ( v28 < *(_QWORD *)(k + 128) + v30 && !*(_BYTE *)(v29 + 137) )
              break;
          }
          v29 = *(_QWORD *)(v29 + 112);
          if ( !v29 )
            goto LABEL_45;
        }
        if ( *(_QWORD *)(v29 + 8) )
        {
          if ( (*(_BYTE *)(v29 + 146) & 8) != 0 && *(_BYTE *)(a10 + 136) && *(_BYTE *)(a10 + 141) )
            goto LABEL_45;
          if ( a5 )
          {
            v47 = v29;
            v35 = *(void (__fastcall **)(const char *))a10;
            if ( v51 )
              ((void (__fastcall *)(char *, __int64))v35)("_", a10);
            else
              ((void (__fastcall *)(char *, __int64))v35)(".", a10);
            if ( *(_BYTE *)(a10 + 136) && *(_BYTE *)(a10 + 141) )
              v51 = (*(_BYTE *)(v47 + 145) & 0x10) != 0;
            sub_74C010(v47, 8, a10);
            v28 = *a8;
            v31 = *(_QWORD *)(v47 + 128);
            v32 = *(_QWORD *)(v47 + 120);
          }
        }
        else if ( *(_BYTE *)(a10 + 136) && *(_BYTE *)(a10 + 141) )
        {
          goto LABEL_45;
        }
        m = v32;
        *a8 = v28 - v31;
      }
      else
      {
        if ( v25 != 11 )
          goto LABEL_45;
        if ( *a8 )
          goto LABEL_45;
        v27 = sub_745C70(j, v24, a3);
        if ( !v27 )
          goto LABEL_45;
        if ( a5 )
        {
          v52 = v27;
          (*(void (__fastcall **)(char *))a10)(".");
          sub_74C010(v52, 8, a10);
          v27 = v52;
        }
        v51 = 0;
        m = *(_QWORD *)(v27 + 120);
      }
      goto LABEL_28;
    }
    v51 = a3 & v18;
    if ( (a3 & v18) != 0 )
      break;
    v36 = sub_8D4050(j);
    for ( m = v36; *(_BYTE *)(v36 + 140) == 12; v36 = *(_QWORD *)(v36 + 160) )
      ;
    v37 = *(_QWORD *)(v36 + 128);
    v38 = *a8;
    if ( !v37 )
      v37 = 1;
    v39 = v38 / v37;
    if ( a5 )
    {
      v46 = v37;
      v48 = v38 / v37;
      (*(void (__fastcall **)(char *, __int64, __int64))a10)("[", a10, v38 % v37);
      sub_7451C0(v48, (__int64 (__fastcall **)(char *, _QWORD))a10);
      (*(void (__fastcall **)(char *))a10)("]");
      v38 = *a8;
      v37 = v46;
      v39 = v48;
    }
    *a8 = v38 - v39 * v37;
LABEL_28:
    v18 |= sub_745A90(m, v24, a3, v56);
    if ( v18 )
    {
      if ( !*a8 )
      {
        v34 = (void *)m;
        *a9 = 1;
        *a7 = v56[0];
        goto LABEL_46;
      }
      v18 = 1;
    }
  }
  v20 = (void *)m;
  *a9 = 1;
  *a7 = 1;
LABEL_12:
  if ( !*a9 )
  {
LABEL_13:
    *a8 = v50;
    goto LABEL_14;
  }
  v54 = v20;
LABEL_49:
  if ( *(_BYTE *)(a1 + 176) )
    v54 = (void *)sub_8D4050(v54);
LABEL_14:
  *a6 = v54;
  return a6;
}
