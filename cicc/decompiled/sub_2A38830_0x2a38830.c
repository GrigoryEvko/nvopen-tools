// Function: sub_2A38830
// Address: 0x2a38830
//
void __fastcall sub_2A38830(__int64 a1, unsigned __int8 *a2, char a3, __int64 a4)
{
  __int64 *v5; // r13
  __int64 *v6; // r15
  __int64 v7; // rsi
  __int8 *v8; // rsi
  char *v9; // r15
  unsigned __int64 v10; // r13
  const char *v11; // rax
  int v12; // ebx
  char *v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  const __m128i *v20; // rbx
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // rdx
  __m128i *v23; // rax
  __m128i v24; // xmm2
  char *v25; // rbx
  char *v26; // [rsp+18h] [rbp-128h]
  char v29; // [rsp+3Eh] [rbp-102h] BYREF
  char v30; // [rsp+3Fh] [rbp-101h] BYREF
  __int64 *v31; // [rsp+40h] [rbp-100h] BYREF
  __int64 v32; // [rsp+48h] [rbp-F8h]
  _BYTE v33[16]; // [rsp+50h] [rbp-F0h] BYREF
  __int128 v34; // [rsp+60h] [rbp-E0h] BYREF
  _QWORD v35[2]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v36; // [rsp+80h] [rbp-C0h]
  _QWORD v37[4]; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 v38; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v39; // [rsp+B8h] [rbp-88h]
  _BYTE v40[128]; // [rsp+C0h] [rbp-80h] BYREF

  v32 = 0x200000000LL;
  v31 = (__int64 *)v33;
  sub_98BCC0((__int64)a2, (__int64)&v31);
  v39 = 0x200000000LL;
  v38 = (unsigned __int64)v40;
  v5 = &v31[(unsigned int)v32];
  v6 = v31;
  if ( v31 == v5 )
    goto LABEL_32;
  do
  {
    v7 = *v6++;
    sub_2A377B0(a1, v7, (__int64)&v38);
  }
  while ( v5 != v6 );
  if ( !(_DWORD)v39 )
  {
LABEL_32:
    v17 = sub_BD4FF0(a2, *(_QWORD *)(a1 + 32), &v29, &v30);
    if ( !v17 )
      goto LABEL_27;
    v35[1] = v17;
    v19 = (unsigned int)v39;
    v20 = (const __m128i *)&v34;
    LOBYTE(v36) = 1;
    v21 = (unsigned int)v39 + 1LL;
    v22 = v38;
    v35[0] = 0;
    v34 = 0;
    if ( v21 > HIDWORD(v39) )
    {
      if ( v38 > (unsigned __int64)&v34 || (unsigned __int64)&v34 >= v38 + 40LL * (unsigned int)v39 )
      {
        sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 0x28u, v21, v18);
        v22 = v38;
        v19 = (unsigned int)v39;
      }
      else
      {
        v25 = (char *)&v35[-2] - v38;
        sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 0x28u, v21, v18);
        v22 = v38;
        v19 = (unsigned int)v39;
        v20 = (const __m128i *)&v25[v38];
      }
    }
    v23 = (__m128i *)(v22 + 40 * v19);
    *v23 = _mm_loadu_si128(v20);
    v24 = _mm_loadu_si128(v20 + 1);
    LODWORD(v39) = v39 + 1;
    v23[1] = v24;
    v23[2].m128i_i64[0] = v20[2].m128i_i64[0];
  }
  v8 = "\n Read Variables: ";
  if ( !a3 )
    v8 = "\n Written Variables: ";
  sub_B18290(a4, v8, a3 == 0 ? 21LL : 18LL);
  if ( (_DWORD)v39 )
  {
    v9 = "RVarName";
    v10 = v38;
    if ( !a3 )
      v9 = "WVarName";
    v11 = "WVarSize";
    if ( a3 )
      v11 = "RVarSize";
    v12 = 0;
    v26 = (char *)v11;
    while ( 1 )
    {
      v13 = "<unknown>";
      v14 = 9;
      if ( *(_BYTE *)(v10 + 16) )
      {
        v13 = *(char **)v10;
        v14 = *(_QWORD *)(v10 + 8);
      }
      sub_B16430((__int64)&v34, v9, 8u, v13, v14);
      sub_2A38130(a4, (__int64)&v34);
      if ( v36 != v37 )
        j_j___libc_free_0((unsigned __int64)v36);
      if ( (_QWORD *)v34 != v35 )
        j_j___libc_free_0(v34);
      if ( !*(_BYTE *)(v10 + 32) )
        goto LABEL_19;
      sub_B18290(a4, " (", 2u);
      sub_B16B10((__int64 *)&v34, v26, 8, *(_QWORD *)(v10 + 24));
      v16 = sub_2A38130(a4, (__int64)&v34);
      sub_B18290(v16, " bytes)", 7u);
      if ( v36 != v37 )
        j_j___libc_free_0((unsigned __int64)v36);
      if ( (_QWORD *)v34 == v35 )
      {
LABEL_19:
        v15 = (unsigned int)(v12 + 1);
        v12 = v15;
        if ( (unsigned int)v15 >= (unsigned int)v39 )
          break;
      }
      else
      {
        j_j___libc_free_0(v34);
        v15 = (unsigned int)(v12 + 1);
        v12 = v15;
        if ( (unsigned int)v15 >= (unsigned int)v39 )
          break;
      }
      v10 = v38 + 40 * v15;
      if ( v12 )
        sub_B18290(a4, ", ", 2u);
    }
  }
  sub_B18290(a4, ".", 1u);
LABEL_27:
  if ( (_BYTE *)v38 != v40 )
    _libc_free(v38);
  if ( v31 != (__int64 *)v33 )
    _libc_free((unsigned __int64)v31);
}
