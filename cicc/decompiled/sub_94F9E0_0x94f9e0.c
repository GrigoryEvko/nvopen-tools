// Function: sub_94F9E0
// Address: 0x94f9e0
//
__int64 __fastcall sub_94F9E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // r15
  __int64 v11; // rax
  __int64 v12; // r10
  unsigned __int64 v13; // rsi
  int v14; // edx
  unsigned int **v15; // r15
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rdi
  int v25; // [rsp+18h] [rbp-108h]
  int v26; // [rsp+18h] [rbp-108h]
  _QWORD *v27; // [rsp+20h] [rbp-100h] BYREF
  __int64 v28; // [rsp+28h] [rbp-F8h]
  _QWORD v29[2]; // [rsp+30h] [rbp-F0h] BYREF
  __m128i *v30; // [rsp+40h] [rbp-E0h]
  __int64 v31; // [rsp+48h] [rbp-D8h]
  __m128i v32; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD *v33; // [rsp+60h] [rbp-C0h]
  __int64 v34; // [rsp+68h] [rbp-B8h]
  _QWORD v35[2]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v36; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v37; // [rsp+88h] [rbp-98h]
  _QWORD v38[2]; // [rsp+90h] [rbp-90h] BYREF
  _QWORD *v39; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v40; // [rsp+A8h] [rbp-78h]
  _QWORD v41[2]; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD v42[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v43; // [rsp+D0h] [rbp-50h] BYREF
  __int16 v44; // [rsp+E0h] [rbp-40h]

  v3 = *a2;
  v27 = v29;
  v28 = 0;
  v4 = *(_QWORD *)(v3 + 40);
  LOBYTE(v29[0]) = 0;
  v5 = sub_BCB120(v4);
  v25 = sub_BCF480(v5, 0, 0, 0);
  v6 = *((_DWORD *)a2 + 2);
  if ( v6 > 3 )
  {
    if ( v6 != 4 )
      goto LABEL_26;
    sub_2241130(&v27, 0, v28, "sys", 3);
  }
  else
  {
    if ( v6 <= 1 )
    {
      if ( (unsigned int)v6 <= 1 )
      {
        sub_2241130(&v27, 0, v28, "cta", 3);
        goto LABEL_5;
      }
LABEL_26:
      sub_91B980("unexpected atomic operation scope.", 0);
    }
    sub_2241130(&v27, 0, v28, "gl", 2);
  }
LABEL_5:
  sub_8FD6D0((__int64)v42, "membar.", &v27);
  if ( v42[1] == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v8 = sub_2241490(v42, ";", 1, v7);
  v30 = &v32;
  if ( *(_QWORD *)v8 == v8 + 16 )
  {
    v32 = _mm_loadu_si128((const __m128i *)(v8 + 16));
  }
  else
  {
    v30 = *(__m128i **)v8;
    v32.m128i_i64[0] = *(_QWORD *)(v8 + 16);
  }
  v9 = *(_QWORD *)(v8 + 8);
  *(_BYTE *)(v8 + 16) = 0;
  v31 = v9;
  *(_QWORD *)v8 = v8 + 16;
  *(_QWORD *)(v8 + 8) = 0;
  if ( (__int64 *)v42[0] != &v43 )
    j_j___libc_free_0(v42[0], v43 + 1);
  v10 = (__int64 *)a2[2];
  strcpy((char *)v35, "~{memory}");
  v39 = v41;
  v33 = v35;
  v34 = 9;
  sub_9486A0((__int64 *)&v39, v35, (__int64)&v35[1] + 1);
  v36 = v38;
  sub_9486A0((__int64 *)&v36, v30, (__int64)v30->m128i_i64 + v31);
  v11 = sub_B41A60(v25, (_DWORD)v36, v37, (_DWORD)v39, v40, 1, 0, 0, 0);
  v12 = *v10;
  v13 = 0;
  v44 = 257;
  v14 = v11;
  v15 = (unsigned int **)(v12 + 48);
  if ( v11 )
  {
    v26 = v11;
    v16 = sub_B3B7D0(v11, 0);
    v14 = v26;
    v13 = v16;
  }
  v17 = sub_921880(v15, v13, v14, 0, 0, (__int64)v42, 0);
  v19 = sub_BD5C60(v17, v13, v18);
  *(_QWORD *)(v17 + 72) = sub_A7A090(v17 + 72, v19, 0xFFFFFFFFLL, 41);
  v21 = sub_BD5C60(v17, v19, v20);
  v22 = sub_A7A090(v17 + 72, v21, 0xFFFFFFFFLL, 6);
  *(_QWORD *)a1 = v17;
  v23 = v36;
  *(_QWORD *)(v17 + 72) = v22;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v23 != v38 )
    j_j___libc_free_0(v23, v38[0] + 1LL);
  if ( v39 != v41 )
    j_j___libc_free_0(v39, v41[0] + 1LL);
  if ( v33 != v35 )
    j_j___libc_free_0(v33, v35[0] + 1LL);
  if ( v30 != &v32 )
    j_j___libc_free_0(v30, v32.m128i_i64[0] + 1);
  if ( v27 != v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  return a1;
}
