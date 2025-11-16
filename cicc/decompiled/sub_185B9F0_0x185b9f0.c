// Function: sub_185B9F0
// Address: 0x185b9f0
//
__int64 __fastcall sub_185B9F0(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // r14d
  __int64 v5; // rbx
  __int64 v6; // rdi
  int v7; // r9d
  _QWORD *v8; // r13
  unsigned __int8 v9; // al
  __int64 v10; // rax
  char v11; // dl
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // r13
  _BYTE *v16; // rdx
  __int64 v17; // r15
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 *v21; // rcx
  __int64 i; // r15
  __int64 *v23; // rax
  __int64 v24; // r8
  unsigned __int8 v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  _BYTE *v29; // rdi
  __int64 v31; // rax
  unsigned __int8 v32; // al
  unsigned __int8 v33; // [rsp+8h] [rbp-248h]
  __int64 v34; // [rsp+8h] [rbp-248h]
  unsigned __int8 v35; // [rsp+8h] [rbp-248h]
  __int64 v36; // [rsp+8h] [rbp-248h]
  _BYTE *v37; // [rsp+10h] [rbp-240h] BYREF
  __int64 v38; // [rsp+18h] [rbp-238h]
  _BYTE v39[560]; // [rsp+20h] [rbp-230h] BYREF

  v3 = 0;
  v5 = *(_QWORD *)(a1 + 8);
  v37 = v39;
  v38 = 0x2000000000LL;
  if ( !v5 )
    return v3;
  do
  {
    while ( 1 )
    {
      v6 = v5;
      v5 = *(_QWORD *)(v5 + 8);
      v8 = sub_1648700(v6);
      v9 = *((_BYTE *)v8 + 16);
      if ( v9 <= 0x17u )
      {
        if ( v9 == 5 )
        {
          if ( !v8[1] )
          {
            v3 = 1;
            sub_159D850((__int64)v8);
          }
        }
        else if ( v9 <= 0x10u )
        {
          v32 = sub_1ACF050(v8);
          if ( v32 )
          {
            v35 = v32;
            sub_159D850((__int64)v8);
            LODWORD(v38) = 0;
            sub_185B9F0(a1, a2);
            v29 = v37;
            v3 = v35;
            goto LABEL_51;
          }
        }
        goto LABEL_6;
      }
      if ( v9 == 55 )
      {
        v24 = *(v8 - 6);
        v25 = *(_BYTE *)(v24 + 16);
        if ( v25 > 0x10u )
        {
LABEL_41:
          if ( v25 > 0x17u )
          {
            v26 = *(_QWORD *)(v24 + 8);
            if ( v26 )
            {
              if ( !*(_QWORD *)(v26 + 8) )
              {
                v27 = (unsigned int)v38;
                if ( (unsigned int)v38 >= HIDWORD(v38) )
                {
                  v36 = v24;
                  sub_16CD150((__int64)&v37, v39, 0, 16, v24, v7);
                  v27 = (unsigned int)v38;
                  v24 = v36;
                }
                v28 = &v37[16 * v27];
                *v28 = v24;
                v28[1] = v8;
                LODWORD(v38) = v38 + 1;
              }
            }
          }
          goto LABEL_6;
        }
        goto LABEL_48;
      }
      if ( v9 != 78 )
        goto LABEL_6;
      v10 = *(v8 - 3);
      if ( *(_BYTE *)(v10 + 16) )
        goto LABEL_6;
      v11 = *(_BYTE *)(v10 + 33);
      if ( (v11 & 0x20) == 0 )
        goto LABEL_6;
      if ( *(_DWORD *)(v10 + 36) != 137 )
        break;
      v24 = v8[3 * (1LL - (*((_DWORD *)v8 + 5) & 0xFFFFFFF))];
      v25 = *(_BYTE *)(v24 + 16);
      if ( v25 > 0x10u )
        goto LABEL_41;
LABEL_48:
      v3 = 1;
      sub_15F20C0(v8);
LABEL_6:
      if ( !v5 )
        goto LABEL_18;
    }
    if ( (v11 & 0x20) == 0 || (*(_DWORD *)(v10 + 36) & 0xFFFFFFFD) != 0x85 )
      goto LABEL_6;
    v12 = sub_1649C60(v8[3 * (1LL - (*((_DWORD *)v8 + 5) & 0xFFFFFFF))]);
    if ( *(_BYTE *)(v12 + 16) != 3 )
      BUG();
    v13 = *(_BYTE *)(v12 + 80) & 1;
    if ( !v13 )
      goto LABEL_6;
    v33 = v13;
    sub_15F20C0(v8);
    v3 = v33;
  }
  while ( v5 );
LABEL_18:
  if ( (_DWORD)v38 )
  {
    v14 = 0;
    v15 = 16 * ((unsigned int)(v38 - 1) + 1LL);
    do
    {
      v16 = &v37[v14];
      v17 = *(_QWORD *)&v37[v14];
      v18 = *(unsigned __int8 *)(v17 + 16);
      if ( (unsigned __int8)v18 <= 0x10u )
      {
LABEL_35:
        sub_15F20C0(*((_QWORD **)v16 + 1));
        for ( i = *(_QWORD *)&v37[v14]; !(unsigned __int8)sub_140AF60(i, a2, 0); i = v34 )
        {
          if ( (*(_BYTE *)(i + 23) & 0x40) != 0 )
          {
            v23 = *(__int64 **)(i - 8);
            v34 = *v23;
            if ( *(_BYTE *)(*v23 + 16) <= 0x17u )
              break;
          }
          else
          {
            v31 = i - 24LL * (*(_DWORD *)(i + 20) & 0xFFFFFFF);
            v34 = *(_QWORD *)v31;
            if ( *(_BYTE *)(*(_QWORD *)v31 + 16LL) <= 0x17u )
              break;
          }
          sub_15F20C0((_QWORD *)i);
        }
        sub_15F20C0((_QWORD *)i);
      }
      else
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v17 + 8);
          if ( !v19 )
            break;
          if ( *(_QWORD *)(v19 + 8) )
            break;
          if ( (unsigned __int8)v18 <= 0x36u )
          {
            v20 = 0x40000020020000LL;
            if ( _bittest64(&v20, v18) )
              break;
          }
          if ( !(unsigned __int8)sub_140AF60(v17, a2, 0) )
          {
            if ( (unsigned __int8)sub_15F3040(v17) || sub_15F3330(v17) )
              break;
            if ( *(_BYTE *)(v17 + 16) == 56 )
            {
              if ( !(unsigned __int8)sub_15FA290(v17) )
                break;
            }
            else if ( (*(_DWORD *)(v17 + 20) & 0xFFFFFFF) != 1 )
            {
              break;
            }
            v21 = (*(_BYTE *)(v17 + 23) & 0x40) != 0
                ? *(__int64 **)(v17 - 8)
                : (__int64 *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
            v17 = *v21;
            v18 = *(unsigned __int8 *)(*v21 + 16);
            if ( (unsigned __int8)v18 > 0x10u )
              continue;
          }
          v16 = &v37[v14];
          goto LABEL_35;
        }
      }
      v14 += 16;
    }
    while ( v15 != v14 );
  }
  v29 = v37;
LABEL_51:
  if ( v29 != v39 )
    _libc_free((unsigned __int64)v29);
  return v3;
}
