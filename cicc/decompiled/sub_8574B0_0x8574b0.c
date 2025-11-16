// Function: sub_8574B0
// Address: 0x8574b0
//
_DWORD *__fastcall sub_8574B0(unsigned __int64 a1, unsigned int *a2)
{
  int v2; // ebx
  bool v3; // r13
  char v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r9
  char *v8; // r12
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  bool v14; // cf
  bool v15; // zf
  __int64 v16; // rcx
  const char *v17; // rdi
  unsigned int *v18; // rsi
  unsigned int v19; // r13d
  bool v20; // cf
  bool v21; // zf
  int v22; // eax
  __m128i *v24; // r14
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  const char *v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rcx

  v2 = a1;
  *a2 = 0;
  if ( (unsigned __int8)sub_7AFE70() <= 1u )
  {
    a1 = 2;
    sub_7AFEC0(2);
  }
  if ( v2 && sub_7B0640() )
  {
    a2 = dword_4F07508;
    a1 = 1662;
    v2 = 0;
    sub_684B30(0x67Eu, dword_4F07508);
  }
  if ( !(unsigned int)sub_857000(a1, a2) )
    sub_685240(0xDu);
  v3 = *qword_4F06410 == 60;
  v4 = v3;
  v8 = (char *)sub_856FF0();
  if ( !v8 )
  {
    a1 = 0;
    v8 = (char *)sub_857470(0);
  }
  v9 = dword_4D04448;
  if ( dword_4D04448 )
  {
    a1 = (unsigned __int64)v8;
    a2 = (unsigned int *)sub_7B06F0(v8, v3, v2, 1);
    if ( a2 )
    {
      a1 = (unsigned __int64)v8;
      if ( sub_7B07A0((__int64)v8, (unsigned __int8 *)a2, v3) )
      {
        v24 = (__m128i *)sub_727790();
        v25 = *(_QWORD *)&dword_4F063F8;
        v24->m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
        v24[1].m128i_i64[0] = v25;
        v26 = sub_727740(1);
        v24[2].m128i_i64[0] = (__int64)v26;
        v26[1] = v8;
        v27 = v24[2].m128i_i64[0];
        *(_BYTE *)(v27 + 40) = *(_BYTE *)(v27 + 40) & 0xFE | v3;
        sub_7B8B50(1u, a2, v27, v28, v29, v30);
        if ( word_4F06418[0] != 10 )
          sub_855DA0(1u, (__int64)a2, v31, v32, v33, v34);
        return sub_824F50(v24);
      }
    }
  }
  sub_7B8B50(a1, a2, v5, v6, v9, v7);
  if ( word_4F06418[0] != 10 )
    sub_855DA0(a1, (__int64)a2, v10, v11, v12, v13);
  dword_4F04D98 = 1;
  v14 = 0;
  v15 = unk_4F068FC == 0;
  if ( unk_4F068FC )
  {
    v16 = 9;
    v17 = "stdarg.h";
    v18 = (unsigned int *)v8;
    do
    {
      if ( !v16 )
        break;
      v14 = *(_BYTE *)v18 < *v17;
      v15 = *(_BYTE *)v18 == *v17;
      v18 = (unsigned int *)((char *)v18 + 1);
      ++v17;
      --v16;
    }
    while ( v15 );
    v19 = (char)((!v14 && !v15) - v14);
    if ( (!v14 && !v15) == v14 )
      goto LABEL_32;
    v20 = dword_4F077C4 < 2u;
    v21 = dword_4F077C4 == 2;
    if ( dword_4F077C4 == 2 )
    {
      v35 = 8;
      v36 = "cstdarg";
      v18 = (unsigned int *)v8;
      do
      {
        if ( !v35 )
          break;
        v20 = *(_BYTE *)v18 < *v36;
        v21 = *(_BYTE *)v18 == *v36;
        v18 = (unsigned int *)((char *)v18 + 1);
        ++v36;
        --v35;
      }
      while ( v21 );
      if ( (!v20 && !v21) == v20 )
      {
        v19 = 1;
LABEL_32:
        if ( !unk_4D04980 )
        {
          sub_885C00(113, "va_start");
          sub_885C00(114, "va_arg");
          sub_885C00(115, "va_end");
          sub_822070("va_start", "va_start", 0, 0);
          sub_822070("va_arg", "va_arg", 0, 0);
          sub_822070("va_end", "va_end", 0, 0);
          v18 = (unsigned int *)dword_4D0429C;
          if ( dword_4D0429C )
          {
            sub_885C00(116, "va_copy");
            v18 = (unsigned int *)"va_copy";
            sub_822070("va_copy", "va_copy", 0, 0);
          }
        }
        sub_8866F0(v19, v18);
        v40 = (unsigned int)dword_4D0493C;
        if ( dword_4D0493C )
        {
          dword_4D03CF4 = 0;
          unk_4D03CF0 = 1;
          while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
            sub_7B8B50(v19, v18, v37, v40, v38, v39);
          unk_4D03CF0 = 0;
        }
        dword_4D03CC0[0] = 1;
        return dword_4D03CC0;
      }
    }
  }
  v22 = dword_4D04944;
  if ( dword_4D04944 )
  {
    v22 = 1;
    if ( HIDWORD(qword_4F077B4) )
      v22 = dword_4D0493C != 0;
  }
  return sub_7B2160(v8, 1, 1u, v4, 0, 0, 0, v2, v22, 0);
}
