// Function: sub_745310
// Address: 0x745310
//
__int64 __fastcall sub_745310(const __m128i *a1, unsigned __int8 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // al
  const __m128i *v5; // r10
  const char *v7; // r13
  int v9; // r14d
  const char *v10; // r12
  char *v11; // rdi
  void (__fastcall *v12)(char *, __int64); // rax
  void (__fastcall *v13)(char *, __int64); // rax
  int v15; // eax
  const char *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rsi
  const char *v22; // rdx
  char *v23; // rdx
  int v24; // ecx
  char *v25; // rdx
  unsigned __int64 v26; // [rsp+18h] [rbp-88h]
  __int64 v27; // [rsp+24h] [rbp-7Ch] BYREF
  int v28; // [rsp+2Ch] [rbp-74h] BYREF
  char s[112]; // [rsp+30h] [rbp-70h] BYREF

  v4 = a2;
  v5 = a1;
  v7 = byte_3F871B3;
  v9 = unk_4F06934;
  v26 = unk_4F068D8;
  if ( *(_BYTE *)(a4 + 137) )
  {
    if ( a2 == 2 )
    {
      (*(void (__fastcall **)(char *, __int64))a4)("(float)", a4);
      v5 = a1;
      v4 = 2;
    }
LABEL_26:
    v10 = byte_3F871B3;
    v11 = sub_70B160(v4, v5, &v27, (_DWORD *)&v27 + 1, &v28);
    if ( !*(_BYTE *)(a4 + 136) )
      return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
    goto LABEL_27;
  }
  v10 = (const char *)&unk_3C15918;
  if ( a2 )
  {
    if ( a2 == 2 )
    {
      v7 = "f";
      v10 = "F";
      v9 = unk_4F06940;
    }
    else
    {
      v7 = "f32x";
      v10 = "f32x";
      if ( a2 != 3 )
      {
        switch ( a2 )
        {
          case 6u:
            v7 = "l";
            v10 = (const char *)&unk_444FD22;
            v9 = unk_4F06928;
            break;
          case 5u:
            v7 = "f64x";
            v10 = "f64x";
            v9 = unk_4F0691C;
            break;
          case 7u:
            v7 = "w";
            v10 = "W";
            v9 = unk_4F0691C;
            break;
          case 8u:
            v7 = "q";
            v10 = (const char *)&unk_3F7C292;
            v9 = unk_4F06910;
            break;
          default:
            v7 = byte_3F871B3;
            v10 = "bf16";
            if ( a2 != 9 )
            {
              v10 = "f16";
              if ( a2 != 10 )
              {
                v10 = "f32";
                if ( a2 != 11 )
                {
                  v10 = "f64";
                  if ( a2 != 12 )
                  {
                    v10 = "f128";
                    if ( a2 != 13 )
                      v10 = byte_3F871B3;
                  }
                }
              }
            }
            break;
        }
      }
    }
  }
  if ( *(_BYTE *)(a4 + 157) )
    goto LABEL_26;
  v11 = sub_70B160(a2, a1, &v27, (_DWORD *)&v27 + 1, &v28);
  if ( !*(_BYTE *)(a4 + 136) )
    goto LABEL_5;
LABEL_27:
  v15 = v28;
  if ( v27 )
  {
    if ( !v28 )
    {
      v16 = "-1.0";
      if ( (_DWORD)v27 )
        v16 = "1.0";
      goto LABEL_53;
    }
LABEL_29:
    if ( a3 && *(_BYTE *)(a3 + 24) == 1 )
    {
      do
      {
        if ( (*(_BYTE *)(a3 + 27) & 2) == 0 )
          break;
        if ( !sub_730740(a3) )
          break;
        a3 = *(_QWORD *)(a3 + 72);
      }
      while ( *(_BYTE *)(a3 + 24) == 1 );
      v15 = v28;
      if ( !v28 )
      {
LABEL_90:
        v16 = "0.0";
        goto LABEL_53;
      }
      if ( unk_4F068D0 )
      {
LABEL_37:
        v16 = "0.0";
        if ( *(_BYTE *)(a3 + 24) == 1 )
        {
LABEL_38:
          v16 = "0.0";
          if ( *(_BYTE *)(a3 + 56) != 105 )
            goto LABEL_53;
          v17 = *(_QWORD *)(a3 + 72);
          if ( *(_BYTE *)(v17 + 24) != 20 )
            goto LABEL_53;
          v18 = *(_QWORD *)(v17 + 56);
          if ( *(_BYTE *)(v18 + 174) )
            goto LABEL_53;
          v19 = *(_QWORD *)(v17 + 16);
          if ( *(_WORD *)(v18 + 176) && v19 )
          {
            if ( *(_BYTE *)(v19 + 24) == 2 )
            {
              v20 = *(_QWORD *)(v19 + 56);
              if ( *(_BYTE *)(v20 + 173) == 6 )
              {
                v21 = *(_QWORD *)(v20 + 184);
                if ( *(_BYTE *)(v21 + 173) == 2 )
                {
                  if ( (*(_BYTE *)(v18 + 89) & 8) != 0 )
                    v22 = *(const char **)(v18 + 24);
                  else
                    v22 = *(const char **)(v18 + 8);
                  sprintf(s, "(%s(\"%s\"))", v22, *(const char **)(v21 + 184));
                  v11 = s;
                  return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
                }
              }
            }
            goto LABEL_53;
          }
          goto LABEL_90;
        }
LABEL_53:
        if ( unk_4F068C0 )
        {
LABEL_66:
          sprintf(s, "(%s%s/(0,0.0%s))", v16, v10, v10);
          v11 = s;
          return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
        }
        if ( unk_4F068D0 )
        {
LABEL_55:
          if ( v15 )
          {
            sprintf(s, "(__builtin_nan%s(\"\"))", v7);
            v11 = s;
          }
          else
          {
            v23 = "-";
            if ( !HIDWORD(v27) )
              v23 = (char *)byte_3F871B3;
            sprintf(s, "(%s__builtin_huge_val%s())", v23, v7);
            v11 = s;
          }
          return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
        }
        v24 = unk_4F068E0;
LABEL_72:
        if ( v24 )
        {
          if ( v26 > 0x765B )
            goto LABEL_55;
          if ( !v15 && v26 > 0x739F )
          {
            v25 = "-";
            if ( !HIDWORD(v27) )
              v25 = (char *)byte_3F871B3;
            sprintf(s, "(%s(__extension__ 0x1.0p%d%s))", v25, (unsigned int)(2 * v9 - 1), v10);
            v11 = s;
            return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
          }
        }
        sprintf(s, "(%s%s/0.0%s)", v16, v10, v10);
        v11 = s;
        return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
      }
    }
    else if ( unk_4F068D0 )
    {
      goto LABEL_67;
    }
    if ( unk_4F068C0 )
    {
      if ( unk_4F068BC > 1899 )
      {
        if ( !a3 || *(_BYTE *)(a3 + 24) != 1 )
          goto LABEL_65;
        goto LABEL_38;
      }
      if ( !unk_4F068E0 || v26 <= 0x765B )
      {
LABEL_65:
        v16 = "0.0";
        goto LABEL_66;
      }
    }
    else
    {
      v24 = unk_4F068E0;
      if ( !unk_4F068E0 || v26 <= 0x765B )
      {
        v16 = "0.0";
        goto LABEL_72;
      }
    }
LABEL_67:
    v16 = "0.0";
    if ( !a3 )
      goto LABEL_53;
    goto LABEL_37;
  }
  if ( v28 )
    goto LABEL_29;
LABEL_5:
  if ( !*v10 )
    return (*(__int64 (__fastcall **)(char *, __int64))a4)(v11, a4);
  v12 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
  if ( !v12 )
    v12 = *(void (__fastcall **)(char *, __int64))a4;
  v12(v11, a4);
  v13 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
  if ( !v13 )
    v13 = *(void (__fastcall **)(char *, __int64))a4;
  return ((__int64 (__fastcall *)(const char *, __int64))v13)(v10, a4);
}
