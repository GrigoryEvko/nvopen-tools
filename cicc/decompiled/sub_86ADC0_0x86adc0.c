// Function: sub_86ADC0
// Address: 0x86adc0
//
void __fastcall sub_86ADC0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  int v3; // r13d
  _QWORD *v4; // rax
  unsigned __int64 v5; // rax
  _BYTE *v6; // rcx
  char v7; // dl
  char v8; // dl
  __int64 v9; // r15
  __int64 v10; // rdi
  char v11; // al
  bool v12; // r14
  _QWORD *v13; // rcx
  char v14; // dl
  __int64 v15; // rsi
  char v16; // dl
  __int64 v17; // rdx
  __int64 v18; // rsi
  char v19; // cl
  char v20; // [rsp+Fh] [rbp-31h]

  v1 = *(_QWORD *)(a1 + 256);
  if ( !v1 )
    return;
  v2 = 0x1000000000008C0LL;
  v3 = 0;
  v4 = 0;
  while ( *(char *)(v1 - 8) < 0 )
  {
    if ( v3 )
    {
      while ( v4 )
      {
        v8 = *((_BYTE *)v4 + 16);
        switch ( v8 )
        {
          case 6:
            v6 = (_BYTE *)v4[3];
            v7 = v6[140];
            if ( (unsigned __int8)(v7 - 9) > 2u && (v7 != 2 || (v6[161] & 8) == 0) )
              goto LABEL_47;
            do
            {
              do
                v4 = (_QWORD *)*v4;
              while ( *((_BYTE *)v4 + 16) != 54 );
            }
            while ( v6 != *(_BYTE **)(v4[3] + 16LL) );
            break;
          case 53:
            v17 = v4[3];
            if ( *(_BYTE *)(v17 + 16) != 6
              || (v18 = *(_QWORD *)(v17 + 24), v19 = *(_BYTE *)(v18 + 140), (unsigned __int8)(v19 - 9) > 2u)
              && (v19 != 2 || (*(_BYTE *)(v18 + 161) & 8) == 0) )
            {
              *(_BYTE *)(v17 + 57) &= ~0x80u;
              goto LABEL_14;
            }
            break;
          case 58:
            break;
          default:
            if ( (unsigned __int8)(v8 - 6) <= 1u || v8 == 11 )
            {
              v6 = (_BYTE *)v4[3];
LABEL_47:
              v6[90] &= ~1u;
              goto LABEL_14;
            }
            goto LABEL_14;
        }
        v4 = (_QWORD *)*v4;
      }
    }
LABEL_14:
    v3 = 0;
    v4 = sub_86A730(v1);
    if ( !v4 )
      return;
LABEL_11:
    v1 = (__int64)v4;
  }
  v5 = *(unsigned __int8 *)(v1 + 16);
  if ( (_BYTE)v5 != 53 )
  {
    if ( (unsigned __int8)v5 <= 0x38u && _bittest64(&v2, v5) )
    {
      if ( (_BYTE)v5 != 56 && (*(_BYTE *)(*(_QWORD *)(v1 + 24) + 90LL) & 1) == 0 )
        v3 = 1;
      v4 = (_QWORD *)sub_86A660((_QWORD **)v1);
      goto LABEL_10;
    }
LABEL_44:
    v4 = *(_QWORD **)v1;
    goto LABEL_10;
  }
  v9 = *(_QWORD *)(v1 + 24);
  v10 = *(_QWORD *)(v9 + 32);
  v11 = *(_BYTE *)(v9 + 16);
  if ( v10 )
  {
    v20 = *(_BYTE *)(v9 + 16);
    sub_734A70(v10);
    v11 = v20;
  }
  v12 = v11 == 11;
  if ( ((v11 - 7) & 0xFB) != 0 )
  {
    if ( v11 != 6 && v11 != 56 )
      goto LABEL_44;
    v12 = 0;
  }
  if ( *(char *)(v9 + 57) >= 0 )
    v3 = 1;
  v4 = (_QWORD *)sub_86A660((_QWORD **)v1);
  if ( dword_4F077C4 != 2 || !v12 )
  {
LABEL_10:
    if ( !v4 )
      return;
    goto LABEL_11;
  }
  if ( v4 )
  {
    v13 = v4;
    while ( 1 )
    {
      v14 = *((_BYTE *)v13 + 16);
      if ( v14 != 58 )
      {
        if ( v14 != 53 )
          goto LABEL_11;
        v15 = v13[3];
        v16 = *(_BYTE *)(v15 + 57);
        if ( (v16 & 8) == 0 )
          goto LABEL_11;
        *(_BYTE *)(v15 + 57) = v16 & 0xF6 | 1;
      }
      v13 = (_QWORD *)*v13;
      if ( !v13 )
        goto LABEL_11;
    }
  }
}
