// Function: sub_D4E110
// Address: 0xd4e110
//
__int64 **__fastcall sub_D4E110(__int64 **a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rcx
  __int64 v6; // r13
  int v7; // r8d
  __int64 v8; // rdi
  _QWORD *v9; // rdx
  int v10; // r8d
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r9
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // eax
  __int64 *v19; // rdx
  __int64 v21; // rax
  unsigned int v22; // esi
  int v23; // edi
  int v24; // edx
  __int64 v25; // r15
  unsigned __int64 v26; // r13
  __int32 v27; // eax
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // eax
  int v31; // r10d
  __int64 v32; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v33; // [rsp+18h] [rbp-1B8h] BYREF
  __m128i v34; // [rsp+20h] [rbp-1B0h] BYREF
  __m128i v35; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 *v36; // [rsp+40h] [rbp-190h] BYREF
  _BYTE *v37; // [rsp+48h] [rbp-188h] BYREF
  __int64 v38; // [rsp+50h] [rbp-180h]
  _BYTE v39[376]; // [rsp+58h] [rbp-178h] BYREF

  v5 = a3[1];
  v6 = *a3;
  v37 = v39;
  v38 = 0x800000000LL;
  v7 = *(_DWORD *)(v5 + 24);
  v8 = *(_QWORD *)(v5 + 8);
  v32 = a2;
  v36 = a3;
  v9 = *(_QWORD **)v6;
  if ( v7 )
  {
    v10 = v7 - 1;
    v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( a2 == *v12 )
    {
LABEL_3:
      v14 = (_QWORD *)v12[1];
      if ( v9 != v14 )
      {
        if ( !v14 )
        {
LABEL_27:
          *a1 = v36;
          a1[1] = (__int64 *)(a1 + 3);
          a1[2] = (__int64 *)0x800000000LL;
          return a1;
        }
        while ( 1 )
        {
          v14 = (_QWORD *)*v14;
          if ( v9 == v14 )
            break;
          if ( !v14 )
          {
            *a1 = v36;
            a1[1] = (__int64 *)(a1 + 3);
            a1[2] = (__int64 *)0x800000000LL;
            goto LABEL_11;
          }
        }
      }
    }
    else
    {
      v30 = 1;
      while ( v13 != -4096 )
      {
        v31 = v30 + 1;
        v11 = v10 & (v30 + v11);
        v12 = (__int64 *)(v8 + 16LL * v11);
        v13 = *v12;
        if ( a2 == *v12 )
          goto LABEL_3;
        v30 = v31;
      }
      if ( v9 )
        goto LABEL_27;
    }
  }
  else if ( v9 )
  {
    *a1 = a3;
    a1[1] = (__int64 *)(a1 + 3);
    a1[2] = (__int64 *)0x800000000LL;
    return a1;
  }
  v35.m128i_i64[0] = a2;
  v35.m128i_i32[2] = 0;
  v16 = (unsigned int)sub_B1C700(v6 + 8, v35.m128i_i64, &v33);
  v18 = v38;
  if ( !(_BYTE)v16 )
  {
    v21 = v33;
    v34.m128i_i64[0] = v33;
    v22 = *(_DWORD *)(v6 + 32);
    v23 = *(_DWORD *)(v6 + 24);
    ++*(_QWORD *)(v6 + 8);
    v24 = v23 + 1;
    if ( 4 * (v23 + 1) >= 3 * v22 )
    {
      sub_B23080(v6 + 8, 2 * v22);
    }
    else
    {
      if ( v22 - *(_DWORD *)(v6 + 28) - v24 > v22 >> 3 )
      {
LABEL_19:
        *(_DWORD *)(v6 + 24) = v24;
        if ( *(_QWORD *)v21 != -4096 )
          --*(_DWORD *)(v6 + 28);
        *(_QWORD *)v21 = v35.m128i_i64[0];
        v25 = v32;
        *(_DWORD *)(v21 + 8) = v35.m128i_i32[2];
        v26 = sub_986580(v25);
        v27 = 0;
        if ( v26 )
          v27 = sub_B46E30(v26);
        v35.m128i_i64[0] = v26;
        v35.m128i_i32[2] = v27;
        v34.m128i_i64[0] = sub_986580(v25);
        v34.m128i_i32[2] = 0;
        sub_D46560((__int64)&v37, &v32, &v34, &v35, v28, v29);
        sub_D4DD40(&v36);
        v18 = v38;
        goto LABEL_9;
      }
      sub_B23080(v6 + 8, v22);
    }
    sub_B1C700(v6 + 8, v35.m128i_i64, &v34);
    v24 = *(_DWORD *)(v6 + 24) + 1;
    v21 = v34.m128i_i64[0];
    goto LABEL_19;
  }
LABEL_9:
  v19 = v36;
  a2 = 0x800000000LL;
  a1[2] = (__int64 *)0x800000000LL;
  *a1 = v19;
  a1[1] = (__int64 *)(a1 + 3);
  if ( v18 )
  {
    a2 = (__int64)&v37;
    sub_D4C3E0((__int64)(a1 + 1), (__int64 *)&v37, (__int64)(a1 + 3), v15, v16, v17);
  }
LABEL_11:
  if ( v37 != v39 )
    _libc_free(v37, a2);
  return a1;
}
