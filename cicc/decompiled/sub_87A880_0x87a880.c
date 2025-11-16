// Function: sub_87A880
// Address: 0x87a880
//
unsigned int __fastcall sub_87A880(_BYTE *src, size_t n, __m128i *a3, __int64 *a4)
{
  __int64 *v7; // rax
  __int64 **v8; // rbx
  unsigned int result; // eax
  __int64 v10; // rbx
  _QWORD *v11; // rax
  void *v12; // rdi
  __int64 v13; // rdx
  char *v14; // [rsp+8h] [rbp-48h]
  _QWORD *v15; // [rsp+18h] [rbp-38h]

  *a3 = _mm_loadu_si128(xmmword_4F06660);
  a3[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a3[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  v7 = a4;
  a3[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  if ( !a4 )
    v7 = (__int64 *)&dword_4F077C8;
  a3->m128i_i64[1] = *v7;
  v8 = (__int64 **)unk_4D04A40;
  if ( unk_4D04A40 )
  {
    while ( 1 )
    {
      if ( v8[3] == (__int64 *)n )
      {
        result = memcmp(v8[2], src, n);
        if ( !result )
          break;
      }
      v8 = (__int64 **)*v8;
      if ( !v8 )
        goto LABEL_10;
    }
    v10 = (__int64)v8[1];
  }
  else
  {
LABEL_10:
    v14 = (char *)sub_7279A0(n + 12);
    v11 = (_QWORD *)sub_823970(32);
    v11[1] = 0;
    v15 = v11;
    *v11 = unk_4D04A40;
    unk_4D04A40 = v11;
    v12 = (void *)sub_823970(n + 1);
    v15[2] = v12;
    memcpy(v12, src, n);
    *(_BYTE *)(v15[2] + n) = 0;
    v15[3] = n;
    v10 = sub_877070(v12, src, v13, v15);
    strcpy(v14, "operator \"\"");
    result = (unsigned int)memcpy(v14 + 11, src, n);
    v14[n + 11] = 0;
    *(_QWORD *)(v10 + 16) = n + 11;
    *(_QWORD *)(v10 + 8) = v14;
    v15[1] = v10;
    if ( *src != 95 )
    {
      result = (unsigned int)qword_4F064B0;
      if ( qword_4F064B0 )
      {
        if ( (qword_4F064B0[11] & 2) == 0 && a4 )
          result = sub_684AA0(4 - (((dword_4F077BC | dword_4D04964) == 0) - 1), 0x9CAu, a4);
      }
    }
  }
  a3->m128i_i64[0] = v10;
  return result;
}
