// Function: sub_A619C0
// Address: 0xa619c0
//
__int64 __fastcall sub_A619C0(__int64 a1, const char *a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v9; // rax
  __int64 *v10; // r12
  __int64 v11; // rsi
  __int64 *v13; // rax
  const __m128i *v15; // [rsp+8h] [rbp-188h]
  const __m128i *v16; // [rsp+8h] [rbp-188h]
  _BYTE v17[112]; // [rsp+10h] [rbp-180h] BYREF
  _QWORD v18[2]; // [rsp+80h] [rbp-110h] BYREF
  __int64 v19; // [rsp+90h] [rbp-100h]
  __int64 v20; // [rsp+98h] [rbp-F8h]
  __int64 v21; // [rsp+A0h] [rbp-F0h]
  __int64 v22; // [rsp+A8h] [rbp-E8h]
  __int64 v23; // [rsp+B0h] [rbp-E0h]
  __int64 v24; // [rsp+B8h] [rbp-D8h]
  __int64 v25; // [rsp+C0h] [rbp-D0h]
  __int64 v26; // [rsp+C8h] [rbp-C8h]
  __int64 v27; // [rsp+D0h] [rbp-C0h]
  __int64 v28; // [rsp+D8h] [rbp-B8h]
  __int64 v29; // [rsp+E0h] [rbp-B0h]
  __int64 v30; // [rsp+E8h] [rbp-A8h]
  __int64 v31; // [rsp+F0h] [rbp-A0h]
  __int64 v32; // [rsp+F8h] [rbp-98h]
  __int64 v33; // [rsp+100h] [rbp-90h]
  __int64 v34; // [rsp+108h] [rbp-88h]
  __int64 v35; // [rsp+110h] [rbp-80h]
  __int64 v36; // [rsp+118h] [rbp-78h]
  char v37; // [rsp+120h] [rbp-70h]
  __int64 v38; // [rsp+128h] [rbp-68h]
  __int64 v39; // [rsp+130h] [rbp-60h]
  __int64 v40; // [rsp+138h] [rbp-58h]
  unsigned int v41; // [rsp+140h] [rbp-50h]
  __int64 v42; // [rsp+148h] [rbp-48h]
  __int64 v43; // [rsp+150h] [rbp-40h]
  __int64 v44; // [rsp+158h] [rbp-38h]

  sub_A54BD0((__int64)v17, a1);
  v18[0] = a4;
  v18[1] = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  if ( a5 == 1 || !a6 )
  {
    v16 = sub_A56340(a3, a1);
    v13 = (__int64 *)sub_22077B0(32);
    v10 = v13;
    if ( v13 )
    {
      v13[3] = a4;
      *v13 = (__int64)off_4979428;
      v13[1] = (__int64)v18;
      v13[2] = (__int64)v16;
    }
  }
  else
  {
    v15 = sub_A56340(a3, a1);
    v9 = sub_22077B0(288);
    v10 = (__int64 *)v9;
    if ( v9 )
    {
      *(_QWORD *)(v9 + 24) = a4;
      *(_QWORD *)(v9 + 8) = v18;
      *(_QWORD *)v9 = off_4979450;
      *(_QWORD *)(v9 + 40) = v9 + 56;
      *(_QWORD *)(v9 + 48) = 0x400000000LL;
      *(_QWORD *)(v9 + 224) = v9 + 248;
      *(_QWORD *)(v9 + 16) = v15;
      *(_DWORD *)(v9 + 32) = 0;
      *(_QWORD *)(v9 + 232) = 0x100000004LL;
      *(_DWORD *)(v9 + 240) = 0;
      *(_BYTE *)(v9 + 244) = 1;
      *(_QWORD *)(v9 + 248) = a2;
      *(_QWORD *)(v9 + 216) = 1;
      *(_QWORD *)(v9 + 280) = v17;
    }
  }
  sub_A5C090((__int64)v17, (__int64)a2, v10);
  if ( (unsigned __int8)(*a2 - 5) <= 0x1Fu && *a2 != 7 && !a5 )
  {
    sub_904010((__int64)v17, " = ");
    sub_A5F6B0((__int64)v17, a2, v10);
  }
  if ( v10 )
    (*(void (__fastcall **)(__int64 *))(*v10 + 16))(v10);
  if ( v42 )
    j_j___libc_free_0(v42, v44 - v42);
  sub_C7D6A0(v39, 16LL * v41, 8);
  if ( v34 )
    j_j___libc_free_0(v34, v36 - v34);
  sub_C7D6A0(v31, 8LL * (unsigned int)v33, 8);
  sub_C7D6A0(v27, 8LL * (unsigned int)v29, 8);
  sub_C7D6A0(v23, 8LL * (unsigned int)v25, 8);
  v11 = 8LL * (unsigned int)v21;
  sub_C7D6A0(v19, v11, 8);
  return sub_A54D10((__int64)v17, v11);
}
