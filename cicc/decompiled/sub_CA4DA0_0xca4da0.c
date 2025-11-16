// Function: sub_CA4DA0
// Address: 0xca4da0
//
__int64 __fastcall sub_CA4DA0(__int64 a1, __int64 a2, __m128i *a3, char a4)
{
  char v5; // dl
  bool v7; // zf
  __m128i *v8; // rsi
  char v9; // dl
  char v10; // al
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __m128i *v16; // r15
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-2F8h]
  __int64 v21; // [rsp+10h] [rbp-2F0h]
  __int64 v22; // [rsp+18h] [rbp-2E8h]
  int v24; // [rsp+20h] [rbp-2E0h]
  __int64 v25; // [rsp+20h] [rbp-2E0h]
  __int64 v26; // [rsp+30h] [rbp-2D0h] BYREF
  char v27; // [rsp+38h] [rbp-2C8h]
  __int64 v28[2]; // [rsp+40h] [rbp-2C0h] BYREF
  __int64 v29; // [rsp+50h] [rbp-2B0h] BYREF
  _OWORD v30[2]; // [rsp+60h] [rbp-2A0h] BYREF
  __int64 v31; // [rsp+80h] [rbp-280h]
  __m128i *v32; // [rsp+90h] [rbp-270h] BYREF
  __int64 v33; // [rsp+98h] [rbp-268h]
  __int64 v34; // [rsp+A0h] [rbp-260h]
  _BYTE v35[264]; // [rsp+A8h] [rbp-258h] BYREF
  __int128 v36; // [rsp+1B0h] [rbp-150h] BYREF
  __int64 v37; // [rsp+1C0h] [rbp-140h]
  _BYTE v38[312]; // [rsp+1C8h] [rbp-138h] BYREF

  v5 = a4;
  v7 = *(_BYTE *)(a2 + 328) == 0;
  v32 = (__m128i *)v35;
  v33 = 0;
  v34 = 256;
  *(_QWORD *)&v36 = v38;
  *((_QWORD *)&v36 + 1) = 0;
  v37 = 256;
  if ( v7 || (*(_BYTE *)(a2 + 320) & 1) != 0 )
  {
    v31 = a3[2].m128i_i64[0];
    v30[0] = _mm_loadu_si128(a3);
    v30[1] = _mm_loadu_si128(a3 + 1);
  }
  else
  {
    sub_CA0EC0((__int64)a3, (__int64)&v36);
    v19 = *(_QWORD *)(a2 + 168);
    LOWORD(v31) = 261;
    *(_QWORD *)&v30[0] = v19;
    *((_QWORD *)&v30[0] + 1) = *(_QWORD *)(a2 + 176);
    sub_C846B0((__int64)v30, (unsigned __int8 **)&v36);
    v5 = a4;
    LOWORD(v31) = 261;
    v30[0] = v36;
  }
  v8 = (__m128i *)v30;
  sub_C83520((__int64)&v26, (__int64)v30, v5, &v32);
  v9 = v27 & 1;
  v10 = (2 * (v27 & 1)) | v27 & 0xFD;
  v27 = v10;
  if ( v9 )
  {
    v27 = v10 & 0xFD;
    v11 = v26;
    v26 = 0;
    *(_QWORD *)&v30[0] = v11 | 1;
    v12 = sub_C64300((__int64 *)v30, (__int64 *)v30);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v12;
    v13 = *(_QWORD *)&v30[0];
    *(_QWORD *)(a1 + 8) = v14;
    if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(v30, (__int64)v30);
  }
  else
  {
    v8 = a3;
    v24 = v26;
    sub_CA0F50(v28, (void **)a3);
    v16 = v32;
    v22 = v28[0];
    v21 = v28[1];
    v20 = v33;
    v17 = sub_22077B0(136);
    if ( v17 )
    {
      LOWORD(v31) = 261;
      *(_QWORD *)v17 = off_4979C68;
      *(_DWORD *)(v17 + 8) = v24;
      v8 = (__m128i *)v30;
      *(_QWORD *)&v30[0] = v22;
      *((_QWORD *)&v30[0] + 1) = v21;
      v25 = v17;
      sub_CA3710(v17 + 16, (void **)v30, 0, 0, 0, 0, 0, 0, 0, 0);
      v17 = v25;
      *(_QWORD *)(v25 + 104) = v25 + 120;
      if ( v16 )
      {
        v8 = v16;
        sub_CA1FB0((__int64 *)(v25 + 104), v16, (__int64)v16->m128i_i64 + v20);
        v17 = v25;
      }
      else
      {
        *(_QWORD *)(v25 + 112) = 0;
        *(_BYTE *)(v25 + 120) = 0;
      }
    }
    v18 = (__int64 *)v28[0];
    *(_QWORD *)a1 = v17;
    *(_BYTE *)(a1 + 16) &= ~1u;
    if ( v18 != &v29 )
    {
      v8 = (__m128i *)(v29 + 1);
      j_j___libc_free_0(v18, v29 + 1);
    }
  }
  if ( (v27 & 2) != 0 )
    sub_C0EC50(&v26);
  if ( (v27 & 1) != 0 && v26 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
  if ( (_BYTE *)v36 != v38 )
    _libc_free(v36, v8);
  if ( v32 != (__m128i *)v35 )
    _libc_free(v32, v8);
  return a1;
}
