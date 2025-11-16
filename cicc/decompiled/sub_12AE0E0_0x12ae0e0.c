// Function: sub_12AE0E0
// Address: 0x12ae0e0
//
__int64 __fastcall sub_12AE0E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // r11
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 *v17; // [rsp+10h] [rbp-110h]
  int v18; // [rsp+28h] [rbp-F8h]
  __int64 v19; // [rsp+28h] [rbp-F8h]
  _QWORD v20[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v21; // [rsp+40h] [rbp-E0h]
  _QWORD *v22; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v23; // [rsp+58h] [rbp-C8h]
  _QWORD v24[2]; // [rsp+60h] [rbp-C0h] BYREF
  __m128i *v25; // [rsp+70h] [rbp-B0h]
  __int64 v26; // [rsp+78h] [rbp-A8h]
  __m128i v27; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD *v28; // [rsp+90h] [rbp-90h]
  __int64 v29; // [rsp+98h] [rbp-88h]
  _QWORD v30[2]; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v31; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v32; // [rsp+B8h] [rbp-68h]
  _QWORD v33[2]; // [rsp+C0h] [rbp-60h] BYREF
  _QWORD *v34; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v35; // [rsp+D8h] [rbp-48h]
  _QWORD v36[8]; // [rsp+E0h] [rbp-40h] BYREF

  v22 = v24;
  v3 = *a2;
  v23 = 0;
  v4 = *(_QWORD *)(v3 + 40);
  LOBYTE(v24[0]) = 0;
  v5 = sub_1643270(v4);
  v18 = sub_1644EA0(v5, 0, 0, 0);
  v6 = *((_DWORD *)a2 + 2);
  if ( v6 > 3 )
  {
    if ( v6 != 4 )
      goto LABEL_24;
    sub_2241130(&v22, 0, v23, "sys", 3);
  }
  else
  {
    if ( v6 <= 1 )
    {
      if ( (unsigned int)v6 <= 1 )
      {
        sub_2241130(&v22, 0, v23, "cta", 3);
        goto LABEL_5;
      }
LABEL_24:
      sub_127B630("unexpected atomic operation scope.", 0);
    }
    sub_2241130(&v22, 0, v23, "gl", 2);
  }
LABEL_5:
  sub_8FD6D0((__int64)&v34, "membar.", &v22);
  if ( v35 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v8 = sub_2241490(&v34, ";", 1, v7);
  v25 = &v27;
  if ( *(_QWORD *)v8 == v8 + 16 )
  {
    v27 = _mm_loadu_si128((const __m128i *)(v8 + 16));
  }
  else
  {
    v25 = *(__m128i **)v8;
    v27.m128i_i64[0] = *(_QWORD *)(v8 + 16);
  }
  v9 = *(_QWORD *)(v8 + 8);
  *(_BYTE *)(v8 + 16) = 0;
  v26 = v9;
  *(_QWORD *)v8 = v8 + 16;
  *(_QWORD *)(v8 + 8) = 0;
  if ( v34 != v36 )
    j_j___libc_free_0(v34, v36[0] + 1LL);
  v10 = (__int64 *)a2[2];
  v28 = v30;
  v17 = v10;
  strcpy((char *)v30, "~{memory}");
  v29 = 9;
  v34 = v36;
  sub_12A72D0((__int64 *)&v34, v30, (__int64)&v30[1] + 1);
  v31 = v33;
  sub_12A72D0((__int64 *)&v31, v25, (__int64)v25->m128i_i64 + v26);
  v11 = sub_15EE570(v18, (_DWORD)v31, v32, (_DWORD)v34, v35, 1, 0, 0);
  v12 = *v17;
  v21 = 257;
  v19 = sub_1285290((__int64 *)(v12 + 48), *(_QWORD *)(*(_QWORD *)v11 + 24LL), v11, 0, 0, (__int64)v20, 0);
  v20[0] = *(_QWORD *)(v19 + 56);
  v13 = sub_16498A0(v19);
  v14 = sub_1563AB0(v20, v13, 0xFFFFFFFFLL, 30);
  v15 = v31;
  *(_DWORD *)(a1 + 8) = 0;
  v20[0] = v14;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)(v19 + 56) = v14;
  *(_QWORD *)a1 = v19;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v15 != v33 )
    j_j___libc_free_0(v15, v33[0] + 1LL);
  if ( v34 != v36 )
    j_j___libc_free_0(v34, v36[0] + 1LL);
  if ( v28 != v30 )
    j_j___libc_free_0(v28, v30[0] + 1LL);
  if ( v25 != &v27 )
    j_j___libc_free_0(v25, v27.m128i_i64[0] + 1);
  if ( v22 != v24 )
    j_j___libc_free_0(v22, v24[0] + 1LL);
  return a1;
}
