// Function: sub_744F60
// Address: 0x744f60
//
int *__fastcall sub_744F60(
        __m128i **a1,
        __int64 a2,
        __int64 a3,
        __m128i *a4,
        unsigned int a5,
        __m128i *a6,
        _DWORD *a7,
        int *a8)
{
  __m128i *v8; // r10
  __int64 v12; // rbx
  int *result; // rax
  __int64 v14; // [rsp-10h] [rbp-C0h]
  __int64 v16; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v17; // [rsp+18h] [rbp-98h] BYREF
  _BYTE v18[84]; // [rsp+20h] [rbp-90h] BYREF
  int v19; // [rsp+74h] [rbp-3Ch]

  v8 = a6;
  if ( a2 )
  {
    v12 = a2;
    if ( (*(_DWORD *)(a2 + 176) & 0x11000) == 0x1000 )
    {
      v16 = 0;
      sub_892150(v18);
      v19 = 1;
      sub_892370(a2, &v17, &v16);
      LODWORD(a2) = 0;
      if ( (*(_BYTE *)(v12 + 89) & 4) != 0 )
        a2 = *(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL);
      sub_744F60((_DWORD)a1, a2, v17, v16, 65792, (unsigned int)v18, (__int64)a7, (__int64)a8);
      v8 = a6;
    }
  }
  result = a8;
  if ( !*a8 )
  {
    if ( a4 )
    {
      *a1 = (__m128i *)sub_744A50(*a1, a4, a3, 0, a7, a5, a8, v8);
      return (int *)v14;
    }
  }
  return result;
}
