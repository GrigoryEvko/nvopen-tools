// Function: sub_5C9ED0
// Address: 0x5c9ed0
//
__int64 __fastcall sub_5C9ED0(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  bool v3; // zf
  char v4; // dl
  __int64 *v5; // r12
  char *v6; // rbx
  char *i; // r14
  char v8; // al
  int v9; // r15d
  __int16 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 *v14; // rax
  int v15; // r14d
  char v16; // al
  char v17; // dl
  __int64 v18; // rax
  __int64 v19; // r14
  char v20; // al
  char v21; // al
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  char v26; // al
  __int64 *v27; // r14
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  char *v31; // rax
  char *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rbx
  char v35; // al
  char *v36; // rax
  char *v37; // [rsp+0h] [rbp-80h]
  __int64 v38; // [rsp+8h] [rbp-78h]
  char *v39; // [rsp+10h] [rbp-70h]
  int v40; // [rsp+18h] [rbp-68h]
  char v41; // [rsp+1Ch] [rbp-64h]
  unsigned int v42; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v43; // [rsp+30h] [rbp-50h] BYREF
  __int64 j; // [rsp+38h] [rbp-48h] BYREF
  __int64 v45; // [rsp+40h] [rbp-40h] BYREF
  __int64 v46[7]; // [rsp+48h] [rbp-38h] BYREF

  result = unk_4F061C8;
  v3 = word_4F06418[0] == 27;
  v4 = *(_BYTE *)(unk_4F061C8 + 36LL);
  *(_BYTE *)(unk_4F061C8 + 36LL) = v4 + 1;
  if ( !v3 )
  {
    if ( *a2 == 40 )
    {
      sub_6851D0(125);
      *(_BYTE *)(a1 + 8) = 0;
      result = unk_4F061C8;
      v4 = *(_BYTE *)(unk_4F061C8 + 36LL) - 1;
    }
    goto LABEL_4;
  }
  v45 = unk_4F063F8;
  if ( *a2 )
  {
    v37 = &a2[*a2 == 63];
  }
  else
  {
    v32 = sub_5C79F0(a1);
    sub_6851F0(1094, v32);
    *(_BYTE *)(a1 + 8) = 0;
    v37 = "(*)";
  }
  v5 = (__int64 *)(a1 + 32);
  v6 = v37 + 1;
  sub_7B8B50();
  LOBYTE(v40) = 0;
  do
  {
    if ( *v6 == 63 )
    {
      v39 = v6 + 2;
      if ( v6[1] != 44 )
        v39 = v6 + 1;
      if ( word_4F06418[0] == 28 )
        goto LABEL_68;
      v41 = 1;
    }
    else
    {
      v39 = v6;
      v41 = v40;
    }
    v40 = sub_869470(&v43);
    if ( v40 )
    {
      LOBYTE(v40) = 0;
      for ( i = v39; ; i = v6 )
      {
        v8 = *i;
        v6 = i + 1;
        if ( *i > 116 )
          goto LABEL_21;
        if ( v8 <= 87 )
        {
          if ( v8 != 40 )
          {
            if ( v8 != 42 )
              goto LABEL_21;
            j = 0;
            v27 = &j;
            v46[0] = 0;
            v28 = (__int64 *)sub_5C9CC0(v46);
            for ( j = (__int64)v28; v28; *v27 = (__int64)v28 )
            {
              do
              {
                v27 = v28;
                v28 = (__int64 *)*v28;
              }
              while ( v28 );
              v28 = (__int64 *)sub_5C9CC0(v46);
            }
            v29 = v46[0];
            if ( word_4F06418[0] != 28 )
            {
              if ( v46[0] )
                goto LABEL_59;
              v29 = j;
              v46[0] = j;
            }
            if ( v29 )
            {
LABEL_59:
              sub_6851C0(1878, v29 + 24);
              *(_BYTE *)(a1 + 8) = 0;
            }
            v30 = sub_7276D0();
            *v27 = v30;
            *(_BYTE *)(v30 + 10) = 0;
            *(_QWORD *)(*v27 + 24) = unk_4F063F8;
            v11 = j;
LABEL_20:
            *v5 = v11;
            goto LABEL_21;
          }
          if ( *(_BYTE *)(a1 + 8) == 108 )
          {
            sub_7BE280(27, 125, 0, 0);
            continue;
          }
        }
        else
        {
          switch ( v8 )
          {
            case 'X':
              v9 = unk_4F063F8;
              v10 = unk_4F063FC;
              v38 = sub_6B96B0();
              if ( *(_BYTE *)(v38 + 24) )
              {
                v11 = sub_7276D0();
                *(_BYTE *)(v11 + 10) = 5;
                v12 = unk_4F061D8;
                *(_DWORD *)(v11 + 24) = v9;
                *(_WORD *)(v11 + 28) = v10;
                *(_QWORD *)(v11 + 32) = v12;
                *(_QWORD *)(v11 + 40) = v38;
              }
              else
              {
                *(_BYTE *)(a1 + 8) = 0;
                v11 = 0;
              }
              goto LABEL_20;
            case 'c':
              v20 = i[1];
              if ( v20 != 116 )
                goto LABEL_40;
              if ( (unsigned int)sub_679C10(1029) )
              {
                v6 = i + 2;
                *v5 = sub_5C7940(a1);
              }
              else
              {
                v20 = i[1];
LABEL_40:
                if ( v20 == 105 || v20 == 116 )
                {
                  j = sub_724DC0();
                  v46[0] = unk_4F063F8;
                  sub_6BA680(j);
                  v21 = *(_BYTE *)(j + 173);
                  if ( v21 )
                  {
                    if ( v21 == 1 || v21 == 12 )
                    {
                      v22 = *(_QWORD *)(j + 144);
                      v23 = sub_7276D0();
                      *(_BYTE *)(v23 + 10) = 3;
                      v24 = v23;
                      *(_QWORD *)(v23 + 24) = v46[0];
                      *(_QWORD *)(v23 + 32) = unk_4F061D8;
                      sub_7296C0(&v42);
                      v25 = sub_73A460(j);
                      *(_QWORD *)(v24 + 40) = v25;
                      if ( v22 && !*(_QWORD *)(v25 + 144) && unk_4F04C50 )
                        sub_72D910(v22, 8, v25);
                      sub_729730(v42);
                      goto LABEL_48;
                    }
                    sub_6851C0(661, v46);
                  }
                  *(_BYTE *)(a1 + 8) = 0;
                  v24 = 0;
LABEL_48:
                  sub_724E30(&j);
                  *v5 = v24;
                  v6 = i + 2;
                  break;
                }
              }
              break;
            case 'n':
              if ( word_4F06418[0] == 1 || (unsigned int)sub_7B80B0(word_4F06418[0]) )
              {
                v18 = sub_7276D0();
                *(_BYTE *)(v18 + 10) = 2;
                v19 = v18;
                *(_QWORD *)(v18 + 24) = unk_4F063F8;
                *(_QWORD *)(v18 + 32) = unk_4F063F0;
                *(_WORD *)(v18 + 8) = word_4F06418[0];
                *(_QWORD *)(v18 + 40) = sub_7C9D40();
                sub_7B8B50();
              }
              else
              {
                v19 = 0;
                sub_6851D0(40);
                *(_BYTE *)(a1 + 8) = 0;
              }
              *v5 = v19;
              break;
            case 's':
              v26 = i[1];
              if ( v26 == 110 )
              {
                v6 = i + 2;
                *v5 = sub_5C8320(a1, 1);
              }
              else if ( v26 == 120 )
              {
                v6 = i + 2;
                *v5 = sub_5C8320(a1, 0);
              }
              break;
            case 't':
              *v5 = sub_5C7940(a1);
              break;
            default:
              break;
          }
        }
LABEL_21:
        v13 = sub_867630(v43, 0);
        if ( v13 )
        {
          if ( !*v5 )
            goto LABEL_26;
          *(_QWORD *)(*v5 + 16) = v13;
          *(_BYTE *)(*v5 + 11) |= 1u;
        }
        v14 = (__int64 *)*v5;
        if ( *v5 )
        {
          do
          {
            v5 = v14;
            v14 = (__int64 *)*v14;
          }
          while ( v14 );
        }
LABEL_26:
        v15 = sub_866C00(v43);
        v16 = *v6;
        if ( *v6 == 43 )
        {
          if ( word_4F06418[0] != 28 )
          {
            v6 = v39;
            v17 = *(_BYTE *)(a1 + 8);
            LOBYTE(v40) = 1;
            v16 = *v39;
            if ( v17 != 108 )
              goto LABEL_28;
            goto LABEL_63;
          }
          v16 = *++v6;
        }
        v17 = *(_BYTE *)(a1 + 8);
        if ( v17 != 108 )
          goto LABEL_28;
LABEL_63:
        if ( v16 == 41 )
        {
          sub_7BE280(28, 18, 0, 0);
          v16 = *++v6;
          v17 = *(_BYTE *)(a1 + 8);
        }
LABEL_28:
        if ( v16 == 63 )
        {
          v41 = 1;
          v16 = *++v6;
        }
        v6 += v16 == 44;
        if ( v17 == 108 )
        {
          if ( *v6 != 40 && word_4F06418[0] == 67 )
          {
LABEL_67:
            if ( *v6 == 41 )
              goto LABEL_68;
          }
        }
        else if ( word_4F06418[0] == 67 )
        {
          goto LABEL_67;
        }
        if ( !v15 )
          goto LABEL_77;
      }
    }
    v6 = v39;
LABEL_77:
    ;
  }
  while ( (unsigned int)sub_7BE800(67) );
  if ( *v6 != 41 && (v41 & 1) == 0 )
  {
    v31 = sub_5C79F0(a1);
    sub_6851A0(1910, &unk_4F063F8, v31);
    *(_BYTE *)(a1 + 8) = 0;
  }
LABEL_68:
  if ( !*(_QWORD *)(a1 + 32) )
  {
    v33 = sub_7276D0();
    *(_BYTE *)(v33 + 10) = 0;
    v34 = v33;
    *(_QWORD *)(v33 + 24) = v45;
    *(_QWORD *)(v33 + 32) = unk_4F063F0;
    v35 = v37[1];
    if ( (unsigned __int8)(v35 - 41) > 1u && v35 != 63 && word_4F06418[0] == 28 && *(_BYTE *)(a1 + 8) )
    {
      v36 = sub_5C79F0(a1);
      sub_6851A0(1833, &v45, v36);
      *(_BYTE *)(a1 + 8) = 0;
    }
    *(_QWORD *)(a1 + 32) = v34;
  }
  sub_7BE280(28, 18, 0, 0);
  result = unk_4F061C8;
  v4 = *(_BYTE *)(unk_4F061C8 + 36LL) - 1;
LABEL_4:
  *(_BYTE *)(result + 36) = v4;
  return result;
}
