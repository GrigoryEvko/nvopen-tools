// Function: sub_888BD0
// Address: 0x888bd0
//
__int64 __fastcall sub_888BD0(char *src, __int64 *a2)
{
  const char *v2; // r14
  unsigned int v3; // ebx
  size_t v4; // rax
  const char *v5; // rdi
  char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  __m128i v17; // xmm6
  __m128i v18; // xmm7
  char *v20; // [rsp+8h] [rbp-D8h]
  int v21; // [rsp+1Ch] [rbp-C4h]
  __int64 v22; // [rsp+20h] [rbp-C0h]
  const char *v23; // [rsp+28h] [rbp-B8h]
  __int16 v24; // [rsp+32h] [rbp-AEh]
  int v25; // [rsp+34h] [rbp-ACh]
  int v26; // [rsp+38h] [rbp-A8h]
  int v27; // [rsp+3Ch] [rbp-A4h]
  __int64 v28; // [rsp+48h] [rbp-98h] BYREF
  _BYTE v29[32]; // [rsp+50h] [rbp-90h] BYREF
  __m128i v30; // [rsp+70h] [rbp-70h] BYREF
  __m128i v31; // [rsp+80h] [rbp-60h] BYREF
  __m128i v32; // [rsp+90h] [rbp-50h] BYREF
  __m128i v33[4]; // [rsp+A0h] [rbp-40h] BYREF

  v2 = src;
  v27 = unk_4F07798;
  v30 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v31 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
  v32 = _mm_loadu_si128(&xmmword_4D04A20);
  v26 = unk_4F04D84;
  v33[0] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  v25 = unk_4D04364;
  v23 = qword_4F06410;
  v22 = qword_4F06408;
  v21 = dword_4F061D8;
  v24 = unk_4F061DC;
  v3 = dword_4F04C3C;
  if ( dword_4D0455C && unk_4D04600 <= 0x30DA3u && strstr(src, "__va_list_tag") )
  {
    v4 = strlen(src);
    v20 = (char *)sub_823970(v4 + 1);
    strcpy(v20, src);
    v5 = v20;
    while ( 1 )
    {
      v6 = strstr(v5, "__va_list_tag");
      v5 = v6;
      if ( !v6 )
        break;
      qmemcpy(v6, "__pgi_tag    ", 13);
    }
    v2 = v20;
  }
  dword_4F04C3C = 1;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 0x10u;
  unk_4F07798 = 1;
  unk_4D04364 = 1;
  sub_7B8190();
  sub_7ADF70((__int64)v29, 0);
  sub_7AE210((__int64)v29);
  sub_7BC000((unsigned __int64)v29, 0, v7, v8, v9, v10);
  sub_7CB300(v2, 0, 0, 0, *a2);
  sub_65CD60(&v28);
  sub_7B8B50((unsigned __int64)&v28, 0, v11, v12, v13, v14);
  sub_7B8260();
  unk_4D04364 = v25;
  unk_4F07798 = v27;
  unk_4F04D84 = v26;
  dword_4F04C3C = v3;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                           & 0xEF
                                                           | (16 * (v3 & 1));
  v15 = _mm_loadu_si128(&v30);
  v16 = _mm_loadu_si128(&v31);
  v17 = _mm_loadu_si128(&v32);
  v18 = _mm_loadu_si128(v33);
  qword_4F06410 = v23;
  *(__m128i *)&qword_4D04A00 = v15;
  *(__m128i *)&word_4D04A10 = v16;
  qword_4F06408 = v22;
  xmmword_4D04A20 = v17;
  unk_4D04A30 = v18;
  dword_4F061D8 = v21;
  unk_4F061DC = v24;
  return v28;
}
