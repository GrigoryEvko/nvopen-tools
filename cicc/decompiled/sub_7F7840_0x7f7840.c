// Function: sub_7F7840
// Address: 0x7f7840
//
__m128i *__fastcall sub_7F7840(char *src, __int8 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  __m128i *v8; // r12
  size_t v9; // rax
  char *v10; // rax
  char v11; // al

  v6 = sub_7259C0(7);
  v6[20] = a3;
  v7 = v6;
  *(_BYTE *)(v6[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v6[21] + 16LL) & 0xFD;
  if ( a4 )
    *(_QWORD *)v6[21] = sub_724EF0(a4);
  v8 = sub_725FD0();
  if ( src )
  {
    v9 = strlen(src);
    v10 = (char *)sub_7E1510(v9 + 1);
    v8->m128i_i64[1] = (__int64)strcpy(v10, src);
  }
  v8[10].m128i_i8[12] = a2;
  v11 = 3;
  if ( (unsigned __int8)a2 > 1u )
    v11 = a2 == 2;
  v8[9].m128i_i64[1] = (__int64)v7;
  v8[12].m128i_i8[1] |= 0x10u;
  v8[5].m128i_i8[8] = (16 * v11) | v8[5].m128i_i8[8] & 0x8F;
  return v8;
}
