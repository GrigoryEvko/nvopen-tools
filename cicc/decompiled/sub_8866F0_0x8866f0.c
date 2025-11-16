// Function: sub_8866F0
// Address: 0x8866f0
//
_QWORD *__fastcall sub_8866F0(int a1)
{
  __int64 v2; // r13
  _QWORD *result; // rax
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __int64 v7; // rcx
  __int64 v8; // r8
  _QWORD *v9; // r15
  char v10; // al
  __m128i v11; // xmm5
  __m128i v12; // xmm6
  __m128i v13; // xmm7
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  char v17; // dl
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // r9
  _QWORD *v22; // r14
  char v23; // dl
  __int64 *v24; // r12
  __m128i v25; // xmm5
  __m128i v26; // xmm6
  __m128i v27; // xmm7
  _QWORD *v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 *v32; // r9
  __int64 v33; // [rsp+0h] [rbp-80h]
  int v34; // [rsp+Ch] [rbp-74h]
  unsigned __int64 v35; // [rsp+10h] [rbp-70h] BYREF
  __int64 v36; // [rsp+18h] [rbp-68h]
  __m128i v37; // [rsp+20h] [rbp-60h]
  __m128i v38; // [rsp+30h] [rbp-50h]
  __m128i v39; // [rsp+40h] [rbp-40h]

  v2 = (__int64)qword_4D04980;
  if ( !qword_4D04980 )
  {
    if ( dword_4F068F8 )
      v2 = qword_4D049B8[11];
    v4 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v5 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v6 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v35 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v37 = v4;
    v36 = *(_QWORD *)&dword_4F077C8;
    v38 = v5;
    v39 = v6;
    sub_878540("va_list", 7u, (__int64 *)&v35);
    v9 = (_QWORD *)(dword_4F068F8
                  ? sub_7D4A40((__int64 *)&v35, v2, 0x20u, v7, v8)
                  : sub_7D4600(qword_4F07288, (__int64 *)&v35, 0x20u, v7, v8));
    if ( v9 && ((v10 = *((_BYTE *)v9 + 80), v10 == 3) || dword_4F077C4 == 2 && (unsigned __int8)(v10 - 4) <= 2u) )
    {
      v34 = 0;
      v33 = v9[11];
    }
    else
    {
      v11 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v12 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v13 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v35 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v37 = v11;
      v36 = *(_QWORD *)&dword_4F077C8;
      v38 = v12;
      v39 = v13;
      sub_878540("__gnuc_va_list", 0xEu, (__int64 *)&v35);
      v16 = sub_7D4600(qword_4F07288, (__int64 *)&v35, 0, v14, v15);
      if ( v16 && ((v17 = *(_BYTE *)(v16 + 80), v17 == 3) || dword_4F077C4 == 2 && (unsigned __int8)(v17 - 4) <= 2u) )
        v33 = *(_QWORD *)(v16 + 88);
      else
        v33 = sub_88A060();
      v18 = dword_4F068F8;
      if ( dword_4F068F8 )
      {
        sub_886510((__int64)&v35);
        sub_8602E0(4u, v2);
        v18 = dword_4F04C64;
      }
      v34 = 1;
      v9 = sub_885B80("va_list", 7u, 3u, v18);
      if ( dword_4F068F8 )
        sub_863FD0((__int64)"va_list", 7, v19, v20, dword_4F068F8, v21);
    }
    if ( unk_4F068F4 )
    {
      v2 = qword_4D049B8[11];
      sub_8602E0(4u, v2);
      sub_886510((__int64)&v35);
      v28 = sub_885A40(v9, 1, (__int64)&v35, dword_4F04C64, 0);
      sub_877E90((__int64)v28, 0, v2);
      sub_863FD0((__int64)v28, 0, v29, v30, v31, v32);
    }
    v22 = sub_7259C0(12);
    v23 = *((_BYTE *)v22 + 143);
    v22[20] = v33;
    *((_BYTE *)v22 + 143) = v23 & 0xCF | (32 * (a1 & 1)) | 0x10;
    sub_7365B0((__int64)v22, 0);
    v9[11] = v22;
    sub_877D80((__int64)v22, v9);
    v22[8] = *(_QWORD *)&dword_4F077C8;
    if ( !dword_4F04C3C )
      sub_8699D0((__int64)v22, 6, 0);
    result = (_QWORD *)v9[11];
    qword_4D04980 = result;
    if ( v34 )
    {
      if ( !dword_4F068F8 )
        return result;
      sub_877E90((__int64)v9, (__int64)v22, v2);
    }
  }
  result = (_QWORD *)dword_4F068F8;
  if ( dword_4F068F8 )
  {
    if ( !(dword_4F600B0 | a1) )
    {
      v24 = (__int64 *)*qword_4D04980;
      sub_6506C0(*qword_4D04980, &dword_4F077C8, 0);
      v25 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v26 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v27 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v35 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v37 = v25;
      v36 = *(_QWORD *)&dword_4F077C8;
      v38 = v26;
      v39 = v27;
      result = sub_885A40(v24, 1, (__int64)&v35, 0, 1);
      dword_4F600B0 = 1;
    }
  }
  return result;
}
