// Function: sub_29C0DC0
// Address: 0x29c0dc0
//
__int64 __fastcall sub_29C0DC0(char *a1, size_t a2)
{
  __m128i *v2; // rax
  const __m128i *v3; // rdx
  __m128i *v4; // rcx
  unsigned int v5; // r12d
  unsigned __int64 v7; // [rsp+0h] [rbp-B0h] BYREF
  __m128i *v8; // [rsp+8h] [rbp-A8h]
  __m128i *v9; // [rsp+10h] [rbp-A0h]
  _QWORD v10[16]; // [rsp+20h] [rbp-90h] BYREF
  char v11; // [rsp+A0h] [rbp-10h] BYREF

  v10[0] = "PassManager";
  v10[2] = "PassAdaptor";
  v10[4] = "AnalysisManagerProxy";
  v10[6] = "PrintFunctionPass";
  v10[8] = "PrintModulePass";
  v10[10] = "BitcodeWriterPass";
  v10[12] = "ThinLTOBitcodeWriterPass";
  v10[1] = 11;
  v10[3] = 11;
  v10[5] = 20;
  v10[7] = 17;
  v10[9] = 15;
  v10[11] = 17;
  v10[13] = 24;
  v10[14] = "VerifierPass";
  v10[15] = 12;
  v8 = 0;
  v2 = (__m128i *)sub_22077B0(0x80u);
  v3 = (const __m128i *)v10;
  v4 = v2 + 8;
  v7 = (unsigned __int64)v2;
  v9 = v2 + 8;
  do
  {
    if ( v2 )
      *v2 = _mm_loadu_si128(v3);
    ++v3;
    ++v2;
  }
  while ( v3 != (const __m128i *)&v11 );
  v8 = v4;
  v5 = sub_E41A50(a1, a2, (__int64 *)&v7);
  if ( v7 )
    j_j___libc_free_0(v7);
  return v5;
}
