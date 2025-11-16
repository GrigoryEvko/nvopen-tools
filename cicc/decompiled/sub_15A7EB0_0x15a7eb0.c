// Function: sub_15A7EB0
// Address: 0x15a7eb0
//
__int64 __fastcall sub_15A7EB0(__int64 a1, void *a2, void *a3, char a4)
{
  unsigned __int64 v4; // rax
  void *v5; // rcx
  void *v6; // rdx
  char *v7; // r8
  char *v8; // rdx
  __m128i v10; // xmm0
  void *s2[2]; // [rsp+0h] [rbp-30h] BYREF
  _BYTE v12[9]; // [rsp+1Fh] [rbp-11h] BYREF

  s2[0] = a2;
  s2[1] = a3;
  v12[0] = a4;
  v4 = sub_16D20C0(s2, v12, 1, 0);
  if ( v4 != -1 )
  {
    v5 = s2[1];
    v6 = (void *)(v4 + 1);
    if ( (void *)(v4 + 1) > s2[1] )
      v6 = s2[1];
    v7 = (char *)((char *)s2[1] - (char *)v6);
    v8 = (char *)s2[0] + (unsigned __int64)v6;
    if ( v4 )
    {
      *(void **)a1 = s2[0];
      if ( v4 > (unsigned __int64)v5 )
        v4 = (unsigned __int64)v5;
      *(_QWORD *)(a1 + 16) = v8;
      *(_QWORD *)(a1 + 24) = v7;
      *(_QWORD *)(a1 + 8) = v4;
      if ( !v7 )
      {
        v7 = (char *)v4;
LABEL_6:
        if ( s2[1] != v7 )
          goto LABEL_7;
        goto LABEL_14;
      }
      if ( v4 )
        return a1;
    }
    else
    {
      *(void **)a1 = s2[0];
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = v8;
      *(_QWORD *)(a1 + 24) = v7;
      if ( !v7 )
        goto LABEL_6;
    }
    sub_16BD130("Expected token before separator in datalayout string", 1);
  }
  *(_QWORD *)(a1 + 16) = 0;
  v10 = _mm_loadu_si128((const __m128i *)s2);
  *(_QWORD *)(a1 + 24) = 0;
  *(__m128i *)a1 = v10;
  v7 = *(char **)(a1 + 8);
  if ( s2[1] != v7 )
    goto LABEL_7;
LABEL_14:
  if ( v7 && memcmp(*(const void **)a1, s2[0], (size_t)v7) )
LABEL_7:
    sub_16BD130("Trailing separator in datalayout string", 1);
  return a1;
}
