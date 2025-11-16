// Function: sub_889BE0
// Address: 0x889be0
//
char __fastcall sub_889BE0(
        char *s,
        unsigned __int16 a2,
        char *a3,
        unsigned __int16 a4,
        unsigned __int8 a5,
        unsigned __int16 a6,
        unsigned __int16 a7,
        char *src)
{
  size_t v11; // rax
  _DWORD *v12; // rax
  __int64 v13; // r9
  bool v14; // zf
  __int64 v15; // r9
  char *v16; // r12
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __m128i v19; // xmm7
  size_t v20; // rax
  __int64 *v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  __int64 v24; // [rsp+10h] [rbp-90h]
  unsigned __int64 v28; // [rsp+30h] [rbp-70h] BYREF
  __int64 v29; // [rsp+38h] [rbp-68h]
  __m128i v30; // [rsp+40h] [rbp-60h]
  __m128i v31; // [rsp+50h] [rbp-50h]
  __m128i v32; // [rsp+60h] [rbp-40h]

  v28 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v30 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v31 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v32 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v29 = *(_QWORD *)&dword_4F077C8;
  v11 = strlen(s);
  sub_878540(s, v11, (__int64 *)&v28);
  v12 = (_DWORD *)v28;
  if ( (*(_DWORD *)(v28 + 72) & 0xFF2000) != 0x92000 || a5 == 9 )
  {
    *(_BYTE *)(v28 + 73) |= 0x20u;
    *((_WORD *)v12 + 38) = a4;
    *((_BYTE *)v12 + 74) = a5;
    if ( dword_4D041F4 && (unsigned int)sub_889000(a5, a4, 0) )
    {
      if ( src )
      {
        v15 = sub_888BD0(src, (__int64 *)&dword_4F077C8);
      }
      else
      {
        v15 = *(_QWORD *)(unk_4D03FB8 + 8LL * a7);
        if ( !v15 )
        {
          v22 = (__int64 *)(unk_4D03FB8 + 8LL * a7);
          v15 = sub_888BD0(off_4AE4E20[a7], (__int64 *)&dword_4F063F8);
          *v22 = v15;
        }
      }
      v23 = v15;
      sub_889970(s, v15, a6, (__int64 *)&v28);
      v13 = v23;
    }
    else
    {
      v13 = 0;
    }
    v12 = &dword_4F077C4;
    if ( dword_4F077C4 != 2 )
    {
      v14 = memcmp(s, "__builtin_", 0xAu) == 0;
      LOBYTE(v12) = !v14;
      if ( v14 && (a5 == 9 || s[10] == 95) )
      {
        v24 = v13;
        LODWORD(v12) = sub_8891F0(a2, a3, 1, a6);
        if ( (_DWORD)v12 )
        {
          v16 = s + 10;
          v17 = _mm_loadu_si128(&xmmword_4F06660[1]);
          v18 = _mm_loadu_si128(&xmmword_4F06660[2]);
          v19 = _mm_loadu_si128(&xmmword_4F06660[3]);
          v28 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
          v30 = v17;
          v31 = v18;
          v32 = v19;
          v29 = *(_QWORD *)&dword_4F077C8;
          v20 = strlen(v16);
          sub_878540(v16, v20, (__int64 *)&v28);
          v12 = (_DWORD *)v28;
          *(_BYTE *)(v28 + 73) |= 0x20u;
          *((_WORD *)v12 + 38) = a4;
          *((_BYTE *)v12 + 74) = a5;
          LOBYTE(v12) = dword_4D041F4;
          if ( dword_4D041F4 )
          {
            LODWORD(v12) = sub_889000(a5, a4, 0);
            if ( (_DWORD)v12 )
              LOBYTE(v12) = sub_889970(v16, v24, a6, (__int64 *)&v28);
          }
        }
      }
    }
  }
  return (char)v12;
}
