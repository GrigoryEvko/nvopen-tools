// Function: sub_29FD4E0
// Address: 0x29fd4e0
//
__int64 __fastcall sub_29FD4E0(__int64 a1, unsigned int a2, __m128i a3)
{
  unsigned int v3; // r13d
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  _DWORD *v13; // r12
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 i; // rdi
  unsigned __int64 v19; // rsi
  void *v21; // [rsp+28h] [rbp-110h] BYREF
  __int64 v22; // [rsp+30h] [rbp-108h]
  __int64 v23[4]; // [rsp+48h] [rbp-F0h] BYREF
  __int16 v24; // [rsp+68h] [rbp-D0h]
  _BYTE *v25; // [rsp+78h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+80h] [rbp-B8h]
  _BYTE v27[32]; // [rsp+88h] [rbp-B0h] BYREF
  __int64 v28; // [rsp+A8h] [rbp-90h]
  __int64 v29; // [rsp+B0h] [rbp-88h]
  __int16 v30; // [rsp+B8h] [rbp-80h]
  __int64 *v31; // [rsp+C0h] [rbp-78h]
  void **v32; // [rsp+C8h] [rbp-70h]
  void **v33; // [rsp+D0h] [rbp-68h]
  __int64 v34; // [rsp+D8h] [rbp-60h]
  int v35; // [rsp+E0h] [rbp-58h]
  __int16 v36; // [rsp+E4h] [rbp-54h]
  char v37; // [rsp+E6h] [rbp-52h]
  __int64 v38; // [rsp+E8h] [rbp-50h]
  __int64 v39; // [rsp+F0h] [rbp-48h]
  void *v40; // [rsp+F8h] [rbp-40h] BYREF
  void *v41; // [rsp+100h] [rbp-38h] BYREF

  v3 = _mm_cvtsi128_si32(a3);
  v4 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v37 = 7;
  v31 = (__int64 *)sub_BD5C60(a1);
  v32 = &v40;
  v33 = &v41;
  v26 = 0x200000000LL;
  v40 = &unk_49DA100;
  v25 = v27;
  v34 = 0;
  v41 = &unk_49DA0B0;
  v5 = *(_QWORD *)(a1 + 40);
  v35 = 0;
  v28 = v5;
  v36 = 512;
  v38 = 0;
  v39 = 0;
  v29 = a1 + 24;
  v30 = 0;
  v6 = *(_QWORD *)sub_B46C60(a1);
  v23[0] = v6;
  if ( v6 && (sub_B96E90((__int64)v23, v6, 1), (v9 = v23[0]) != 0) )
  {
    v10 = (unsigned __int64)v25;
    v11 = v26;
    v12 = &v25[16 * (unsigned int)v26];
    if ( v25 != (_BYTE *)v12 )
    {
      while ( *(_DWORD *)v10 )
      {
        v10 += 16LL;
        if ( v12 == (_QWORD *)v10 )
          goto LABEL_21;
      }
      *(_QWORD *)(v10 + 8) = v23[0];
      goto LABEL_8;
    }
LABEL_21:
    if ( (unsigned int)v26 >= (unsigned __int64)HIDWORD(v26) )
    {
      v19 = (unsigned int)v26 + 1LL;
      if ( HIDWORD(v26) < v19 )
      {
        sub_C8D5F0((__int64)&v25, v27, v19, 0x10u, v7, v8);
        v12 = &v25[16 * (unsigned int)v26];
      }
      *v12 = 0;
      v12[1] = v9;
      v9 = v23[0];
      LODWORD(v26) = v26 + 1;
    }
    else
    {
      if ( v12 )
      {
        *(_DWORD *)v12 = 0;
        v12[1] = v9;
        v11 = v26;
        v9 = v23[0];
      }
      LODWORD(v26) = v11 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v25, 0);
    v9 = v23[0];
  }
  if ( v9 )
LABEL_8:
    sub_B91220((__int64)v23, v9);
  v13 = sub_C33310();
  sub_C3B170((__int64)v23, _mm_cvtsi32_si128(v3));
  sub_C407B0(&v21, v23, v13);
  sub_C338F0((__int64)v23);
  v14 = sub_AC8EA0(v31, (__int64 *)&v21);
  if ( v21 == sub_C33340() )
  {
    if ( v22 )
    {
      for ( i = v22 + 24LL * *(_QWORD *)(v22 - 8); v22 != i; i -= 24 )
        sub_91D830((_QWORD *)(i - 24));
      j_j_j___libc_free_0_0(i - 8);
    }
  }
  else
  {
    sub_C338F0((__int64)&v21);
  }
  v15 = *(_QWORD *)(v4 + 8);
  if ( *(_BYTE *)(v15 + 8) != 2 )
    v14 = sub_AA93C0(0x2Eu, v14, v15);
  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v28 + 72), 72) )
    LOBYTE(v36) = 1;
  HIDWORD(v21) = 0;
  v24 = 257;
  v16 = sub_B35C90((__int64)&v25, a2, v4, v14, (__int64)v23, 0, (unsigned int)v21, 0);
  nullsub_61();
  v40 = &unk_49DA100;
  nullsub_63();
  if ( v25 != v27 )
    _libc_free((unsigned __int64)v25);
  return v16;
}
