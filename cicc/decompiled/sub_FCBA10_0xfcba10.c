// Function: sub_FCBA10
// Address: 0xfcba10
//
__int64 __fastcall sub_FCBA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rdx
  __m128i *v12; // rbx
  unsigned __int8 v13; // al
  unsigned int v14; // r12d
  __int64 v15; // rcx
  __int64 v16; // r15
  __int8 *v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // r12
  _QWORD *v22; // rbx
  __int64 v23; // rdi
  __int64 v25; // [rsp+8h] [rbp-4A8h]
  __int64 v26; // [rsp+28h] [rbp-488h]
  __int64 v27; // [rsp+40h] [rbp-470h] BYREF
  _BYTE *v28; // [rsp+48h] [rbp-468h]
  __int64 v29; // [rsp+50h] [rbp-460h]
  _BYTE v30[128]; // [rsp+58h] [rbp-458h] BYREF
  __int64 v31; // [rsp+D8h] [rbp-3D8h]
  __int64 v32; // [rsp+E0h] [rbp-3D0h]
  _QWORD *v33; // [rsp+E8h] [rbp-3C8h] BYREF
  unsigned int v34; // [rsp+F0h] [rbp-3C0h]
  _QWORD v35[2]; // [rsp+3E8h] [rbp-C8h] BYREF
  _BYTE v36[184]; // [rsp+3F8h] [rbp-B8h] BYREF

  v28 = v30;
  v29 = 0x1000000000LL;
  v6 = &v33;
  v27 = a1;
  v31 = 0;
  v32 = 1;
  do
  {
    *v6 = -4096;
    v6 += 3;
  }
  while ( v6 != v35 );
  v35[0] = v36;
  v35[1] = 0x1000000000LL;
  if ( (*(_BYTE *)(a2 + 1) & 0x7F) != 0 )
    v25 = sub_FC83A0((__int64)&v27, a2);
  else
    v25 = (__int64)sub_FC9A50(&v27, (_QWORD *)a2, a3, (__int64)v35, a5, a6);
LABEL_5:
  v10 = (unsigned int)v29;
  while ( (_DWORD)v10 )
  {
    v11 = (unsigned int)v10;
    v10 = (unsigned int)(v10 - 1);
    v12 = *(__m128i **)&v28[8 * v11 - 8];
    LODWORD(v29) = v10;
    v13 = v12[-1].m128i_u8[0];
    v7 = (v13 & 2) != 0;
    if ( (v13 & 2) != 0 )
      v14 = v12[-2].m128i_u32[2];
    else
      v14 = ((unsigned __int16)v12[-1].m128i_i16[0] >> 6) & 0xF;
    if ( v14 )
    {
      v15 = v14;
      v16 = 0;
      v26 = v14;
      while ( 1 )
      {
        if ( (_BYTE)v7 )
        {
          v17 = (__int8 *)v12[-2].m128i_i64[0];
        }
        else
        {
          v7 = -16 - 8LL * ((v13 >> 2) & 0xF);
          v17 = &v12->m128i_i8[v7];
        }
        v18 = *(_QWORD **)&v17[8 * v16];
        a2 = (__int64)v18;
        v19 = sub_FC99C0(&v27, (__int64)v18, v7, v15, v8, v9);
        if ( (_BYTE)v20 )
        {
          v7 = v19;
        }
        else
        {
          a2 = (__int64)v18;
          v7 = (__int64)sub_FC9A50(&v27, v18, v20, v15, v8, v9);
        }
        if ( v18 != (_QWORD *)v7 )
        {
          a2 = (unsigned int)v16;
          sub_BA6610(v12, v16, (unsigned __int8 *)v7);
        }
        if ( v26 == ++v16 )
          break;
        v13 = v12[-1].m128i_u8[0];
        v7 = (v13 & 2) != 0;
      }
      goto LABEL_5;
    }
  }
  if ( (_BYTE *)v35[0] != v36 )
    _libc_free(v35[0], a2);
  if ( (v32 & 1) != 0 )
  {
    v22 = v35;
    v21 = &v33;
  }
  else
  {
    v7 = v34;
    v21 = v33;
    a2 = 24LL * v34;
    if ( !v34 )
      goto LABEL_37;
    v22 = (_QWORD *)((char *)v33 + a2);
    if ( (_QWORD *)((char *)v33 + a2) == v33 )
      goto LABEL_37;
  }
  do
  {
    if ( *v21 != -8192 && *v21 != -4096 )
    {
      v23 = v21[2];
      if ( v23 )
        sub_BA65D0(v23, a2, v7, v10, v8);
    }
    v21 += 3;
  }
  while ( v22 != v21 );
  if ( (v32 & 1) == 0 )
  {
    v21 = v33;
    a2 = 24LL * v34;
LABEL_37:
    sub_C7D6A0((__int64)v21, a2, 8);
  }
  if ( v28 != v30 )
    _libc_free(v28, a2);
  return v25;
}
