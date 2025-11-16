// Function: sub_305FD50
// Address: 0x305fd50
//
__int64 __fastcall sub_305FD50(
        __int64 a1,
        int a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        int a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        int a10,
        char a11)
{
  char v14; // r15
  const char *v15; // rsi
  const char *v16; // r8
  size_t v17; // rax
  const char *v18; // r8
  size_t v19; // r9
  _QWORD *v20; // rdx
  __int64 v21; // rax
  bool v22; // zf
  _BOOL4 v23; // eax
  __int64 v25; // rax
  _QWORD *v26; // rdi
  const char *v27; // [rsp-8h] [rbp-208h]
  size_t n; // [rsp+0h] [rbp-200h]
  const char *src; // [rsp+8h] [rbp-1F8h]
  int v32; // [rsp+28h] [rbp-1D8h]
  int v33; // [rsp+2Ch] [rbp-1D4h]
  __m128i v35; // [rsp+40h] [rbp-1C0h]
  size_t v38; // [rsp+78h] [rbp-188h] BYREF
  _QWORD *v39; // [rsp+80h] [rbp-180h] BYREF
  size_t v40; // [rsp+88h] [rbp-178h]
  _QWORD v41[2]; // [rsp+90h] [rbp-170h] BYREF
  __int64 v42[2]; // [rsp+A0h] [rbp-160h] BYREF
  _QWORD v43[28]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v44; // [rsp+190h] [rbp-70h]

  v35 = _mm_loadu_si128(&a7);
  v14 = a11;
  v32 = a10;
  if ( BYTE4(a9) )
  {
    v33 = a9;
    if ( !(_DWORD)a9 )
      sub_C64ED0("Target does not support the tiny CodeModel", 0);
    if ( (_DWORD)a9 == 2 )
      sub_C64ED0("Target does not support the kernel CodeModel", 0);
  }
  else
  {
    v33 = 1;
  }
  sub_305AFE0(v42, a3, (int)a4, a5, (int)byte_3F871B3, 0, v35.m128i_i64[0], v35.m128i_i64[1]);
  v15 = v27;
  if ( v14 )
  {
    if ( (v44 & 0x100000) != 0 )
    {
      if ( (v44 & 0x200000) != 0 )
        v16 = off_4C5D060[0];
      else
        v16 = off_4C5D090[0];
    }
    else if ( (v44 & 0x200000) != 0 )
    {
      v16 = off_4C5D068[0];
    }
    else
    {
      v16 = off_4C5D098[0];
    }
  }
  else
  {
    v16 = off_4C5D088[0];
  }
  v39 = v41;
  if ( !v16 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  src = v16;
  v17 = strlen(v16);
  v18 = src;
  v38 = v17;
  v19 = v17;
  if ( v17 > 0xF )
  {
    n = v17;
    v25 = sub_22409D0((__int64)&v39, &v38, 0);
    v18 = src;
    v19 = n;
    v39 = (_QWORD *)v25;
    v26 = (_QWORD *)v25;
    v41[0] = v38;
  }
  else
  {
    if ( v17 == 1 )
    {
      LOBYTE(v41[0]) = *src;
      v20 = v41;
      goto LABEL_16;
    }
    if ( !v17 )
    {
      v20 = v41;
      goto LABEL_16;
    }
    v26 = v41;
  }
  v15 = v18;
  memcpy(v26, v18, v19);
  v17 = v38;
  v20 = v39;
LABEL_16:
  v40 = v17;
  *((_BYTE *)v20 + v17) = 0;
  v42[0] = (__int64)&unk_4A303E0;
  sub_35DE000(v42, v15);
  sub_34CF650(a1, a2, (_DWORD)v39, v40, a3, a6, (__int64)a4, a5, v35.m128i_i64[0], v35.m128i_i64[1], 1, v33, v32);
  if ( v39 != v41 )
    j_j___libc_free_0((unsigned __int64)v39);
  *(_BYTE *)(a1 + 1264) = v14;
  *(_QWORD *)a1 = &unk_4A30900;
  sub_305FCF0(v42);
  v21 = v42[0];
  v42[0] = (__int64)v43;
  *(_QWORD *)(a1 + 1272) = v21;
  sub_305CFF0(v42, v35.m128i_i64[0], v35.m128i_i64[0] + v35.m128i_i64[1]);
  v39 = v41;
  sub_305CFF0((__int64 *)&v39, a4, (__int64)&a4[a5]);
  sub_305B3C0(a1 + 1288, a3, (__int64)&v39, v42, a1);
  if ( v39 != v41 )
    j_j___libc_free_0((unsigned __int64)v39);
  if ( (_QWORD *)v42[0] != v43 )
    j_j___libc_free_0(v42[0]);
  *(_QWORD *)(a1 + 539312) = 0;
  *(_QWORD *)(a1 + 539328) = a1 + 539344;
  *(_QWORD *)(a1 + 539336) = 0x400000000LL;
  *(_QWORD *)(a1 + 539376) = a1 + 539392;
  *(_QWORD *)(a1 + 539408) = a1 + 539312;
  *(_QWORD *)(a1 + 539320) = 0;
  *(_QWORD *)(a1 + 539384) = 0;
  *(_QWORD *)(a1 + 539392) = 0;
  *(_QWORD *)(a1 + 539400) = 1;
  *(_QWORD *)(a1 + 539416) = 0;
  v22 = *(_DWORD *)(a3 + 44) == 21;
  *(_QWORD *)(a1 + 539424) = 0;
  v23 = !v22;
  v22 = (_BYTE)qword_502CB88 == 0;
  *(_QWORD *)(a1 + 539432) = 0;
  *(_DWORD *)(a1 + 539440) = 0;
  *(_DWORD *)(a1 + 1280) = v23;
  if ( v22 )
    *(_BYTE *)(a1 + 688) |= 1u;
  return sub_34CF150(a1);
}
