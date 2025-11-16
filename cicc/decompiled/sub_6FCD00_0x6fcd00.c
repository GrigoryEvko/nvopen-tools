// Function: sub_6FCD00
// Address: 0x6fcd00
//
__int64 __fastcall sub_6FCD00(unsigned __int16 a1, __m128i *a2, __m128i *a3, _DWORD *a4, _QWORD *a5, _BYTE *a6)
{
  __int64 i; // r12
  __int64 j; // r15
  int v10; // eax
  int v12; // eax
  __int64 v13; // rdi
  int v14; // eax
  int v15; // r8d
  int v16; // eax
  _BOOL4 v17; // eax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  _QWORD *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 *v28; // rax
  int v30; // [rsp+Ch] [rbp-54h]
  int v31; // [rsp+Ch] [rbp-54h]
  int v32; // [rsp+Ch] [rbp-54h]
  char v33; // [rsp+Ch] [rbp-54h]
  int v34; // [rsp+Ch] [rbp-54h]
  __int64 v37[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = a2->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = a3->m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  switch ( a1 )
  {
    case '"':
    case '\'':
    case '9':
    case ':':
      v31 = sub_8D2B20(i);
      v12 = sub_8D2B20(j);
      if ( !(v12 | v31) )
        goto LABEL_7;
      if ( !v31 )
      {
        v32 = v12;
        if ( !v12 )
          goto LABEL_7;
        v14 = sub_8D2AC0(i);
        v15 = v32;
        if ( !v14 )
        {
          if ( !(unsigned int)sub_8D2930(i) )
            goto LABEL_7;
          LODWORD(j) = *(unsigned __int8 *)(j + 160);
          v13 = (unsigned int)j;
          v33 = j;
          *a5 = sub_72C7D0((unsigned int)j);
          if ( a1 != 34 )
          {
            if ( a1 != 57 )
            {
              if ( a1 == 58 )
                goto LABEL_15;
              if ( a1 != 39 )
LABEL_64:
                sub_721090(v13);
LABEL_40:
              if ( (unsigned int)sub_8D2AC0(i) )
              {
                *a6 = 45;
                goto LABEL_24;
              }
LABEL_58:
              *a6 = 42;
              goto LABEL_24;
            }
            goto LABEL_65;
          }
LABEL_45:
          *a6 = 41;
          goto LABEL_24;
        }
LABEL_53:
        v34 = v15;
        LOBYTE(j) = sub_6E55D0(*(unsigned __int8 *)(i + 160), *(unsigned __int8 *)(j + 160));
        v13 = (unsigned __int8)j;
        *a5 = sub_72C7D0((unsigned __int8)j);
        if ( a1 == 57 )
          goto LABEL_51;
        if ( a1 == 58 )
          goto LABEL_15;
        if ( a1 != 34 )
        {
          if ( a1 != 39 )
            goto LABEL_64;
          if ( !v34 )
            goto LABEL_58;
          goto LABEL_40;
        }
        goto LABEL_45;
      }
      if ( !v12 )
      {
        v16 = sub_8D2AC0(j);
        v15 = 0;
        if ( !v16 )
        {
          if ( (unsigned int)sub_8D2930(j) )
          {
            LOBYTE(j) = *(_BYTE *)(i + 160);
            v13 = (unsigned __int8)j;
            v33 = j;
            *a5 = sub_72C7D0((unsigned __int8)j);
            if ( a1 == 34 )
              goto LABEL_45;
            if ( a1 != 57 )
            {
              if ( a1 == 58 )
                goto LABEL_15;
              if ( a1 != 39 )
                goto LABEL_64;
              goto LABEL_58;
            }
LABEL_65:
            LOBYTE(j) = v33;
            goto LABEL_51;
          }
LABEL_7:
          *a6 = 120;
          return 0;
        }
        goto LABEL_53;
      }
      LOBYTE(j) = sub_6E55D0(*(unsigned __int8 *)(i + 160), *(unsigned __int8 *)(j + 160));
      v13 = (unsigned __int8)j;
      *a5 = sub_72C610((unsigned __int8)j);
      if ( a1 == 39 )
        goto LABEL_40;
      if ( a1 > 0x27u )
      {
        if ( a1 != 57 )
        {
          if ( a1 == 58 )
          {
LABEL_15:
            *a6 = 77;
LABEL_16:
            sub_6FC450(a3, j);
            return 1;
          }
          goto LABEL_64;
        }
        v17 = sub_6E53E0(5, 0x41Au, a4);
        v21 = (__int64)a4;
        if ( v17 )
          sub_684B30(0x41Au, a4);
        v22 = (_QWORD *)sub_6F6F40(a3, 0, v18, v21, v19, v20);
        v37[0] = sub_724DC0(a3, 0, v23, v24, v25, v26);
        sub_72BB40(*v22, v37[0]);
        v27 = sub_73A720(v37[0]);
        v28 = (__int64 *)sub_73DF90(v22, v27);
        sub_6E70E0(v28, (__int64)a3);
        sub_724E30(v37);
LABEL_51:
        *a6 = 76;
        goto LABEL_16;
      }
      if ( a1 != 34 )
        goto LABEL_64;
      *a6 = 44;
LABEL_24:
      sub_6FC450(a2, j);
      sub_6FC450(a3, j);
      return 1;
    case '#':
    case '$':
    case '<':
    case '=':
      v30 = sub_8D2B20(i);
      v10 = sub_8D2B20(j);
      if ( !(v10 | v30) )
        goto LABEL_7;
      if ( v30 && v10 )
      {
        LOBYTE(j) = sub_6E55D0(*(unsigned __int8 *)(i + 160), *(unsigned __int8 *)(j + 160));
        v13 = (unsigned __int8)j;
        *a5 = sub_72C7D0((unsigned __int8)j);
        switch ( a1 )
        {
          case '<':
            *a6 = 74;
            goto LABEL_16;
          case '=':
            *a6 = 75;
            goto LABEL_16;
          case '#':
            *a6 = 39;
            break;
          case '$':
            *a6 = 40;
            break;
          default:
            goto LABEL_64;
        }
        goto LABEL_24;
      }
      if ( (unsigned __int16)(a1 - 35) <= 1u )
      {
        if ( v30 )
        {
          if ( (unsigned int)sub_8D2AC0(j) )
          {
            LOBYTE(j) = sub_6E55D0(*(unsigned __int8 *)(i + 160), *(unsigned __int8 *)(j + 160));
            *a5 = sub_72C6F0((unsigned __int8)j);
            *a6 = 2 * (a1 != 35) + 47;
            goto LABEL_24;
          }
        }
        else if ( v10 && (unsigned int)sub_8D2AC0(i) )
        {
          LOBYTE(j) = sub_6E55D0(*(unsigned __int8 *)(i + 160), *(unsigned __int8 *)(j + 160));
          *a5 = sub_72C6F0((unsigned __int8)j);
          *a6 = 2 * (a1 != 35) + 46;
          goto LABEL_24;
        }
      }
      goto LABEL_7;
    default:
      goto LABEL_7;
  }
}
