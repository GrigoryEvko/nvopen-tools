// Function: sub_2FA7690
// Address: 0x2fa7690
//
__int64 __fastcall sub_2FA7690(__int64 a1, __m128i *a2)
{
  const char *v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int16 v10; // ax
  __int64 *v11; // r14
  unsigned int v12; // r15d
  unsigned __int64 *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int64 v18; // rbx
  _QWORD *v19; // r12
  void (__fastcall *v20)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _QWORD *v28; // r15
  void (__fastcall *v29)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v30; // rax
  unsigned __int64 v31; // [rsp+8h] [rbp-308h]
  unsigned __int64 v32; // [rsp+18h] [rbp-2F8h]
  unsigned __int64 v33[94]; // [rsp+20h] [rbp-2F0h] BYREF

  memset(v33, 0, 0x2C0u);
  v3 = (const char *)&unk_4F8144C;
  v4 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v4
    && (v3 = (const char *)&unk_4F8144C,
        (v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_4F8144C)) != 0) )
  {
    v31 = v9 + 176;
    if ( LOBYTE(v33[87]) )
    {
      LOBYTE(v33[87]) = 0;
      sub_FFCE90((__int64)v33, (__int64)&unk_4F8144C, v5, v6, v7, v8);
      sub_FFD870((__int64)v33, (__int64)&unk_4F8144C, v24, v25, v26, v27);
      sub_FFBC40((__int64)v33, (__int64)&unk_4F8144C);
      v28 = (_QWORD *)v33[84];
      v32 = v33[85];
      if ( v33[85] != v33[84] )
      {
        do
        {
          *v28 = &unk_49E5048;
          v29 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v28[7];
          if ( v29 )
          {
            v3 = (const char *)(v28 + 5);
            v29(v28 + 5, v28 + 5, 3);
          }
          *v28 = &unk_49DB368;
          v30 = v28[3];
          LOBYTE(v6) = v30 != 0;
          LOBYTE(v5) = v30 != -4096;
          if ( ((unsigned __int8)v5 & (v30 != 0)) != 0 && v30 != -8192 )
            sub_BD60C0(v28 + 1);
          v28 += 9;
        }
        while ( (_QWORD *)v32 != v28 );
        v28 = (_QWORD *)v33[84];
      }
      if ( v28 )
      {
        v3 = (const char *)(v33[86] - (_QWORD)v28);
        j_j___libc_free_0((unsigned __int64)v28);
      }
      if ( !BYTE4(v33[74]) )
        _libc_free(v33[72]);
      if ( (unsigned __int64 *)v33[0] != &v33[2] )
        _libc_free(v33[0]);
    }
    LOBYTE(v33[87]) = 1;
    v33[1] = 0x1000000000LL;
    v33[66] = 0;
    v33[68] = v31;
    v33[72] = (unsigned __int64)&v33[75];
    v33[67] = 0;
    v33[69] = 0;
    LOBYTE(v33[70]) = 1;
    v33[71] = 0;
    v33[73] = 8;
    LODWORD(v33[74]) = 0;
    BYTE4(v33[74]) = 1;
    memset(&v33[84], 0, 24);
    LOWORD(v33[83]) = 0;
    v10 = a2->m128i_i16[1];
    v33[0] = (unsigned __int64)&v33[2];
    v11 = (__int64 *)(a1 + 176);
  }
  else
  {
    v12 = LOBYTE(v33[87]);
    v10 = a2->m128i_i16[1];
    v11 = (__int64 *)(a1 + 176);
    if ( !LOBYTE(v33[87]) )
    {
      v13 = 0;
      if ( (v10 & 0x4000) == 0 )
        return v12;
      goto LABEL_23;
    }
  }
  v12 = 0;
  v13 = v33;
  if ( (v10 & 0x4000) == 0 )
  {
LABEL_6:
    LOBYTE(v33[87]) = 0;
    sub_FFCE90((__int64)v33, (__int64)v3, v5, v6, v7, v8);
    sub_FFD870((__int64)v33, (__int64)v3, v14, v15, v16, v17);
    sub_FFBC40((__int64)v33, (__int64)v3);
    v18 = v33[85];
    v19 = (_QWORD *)v33[84];
    if ( v33[85] != v33[84] )
    {
      do
      {
        v20 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v19[7];
        *v19 = &unk_49E5048;
        if ( v20 )
          v20(v19 + 5, v19 + 5, 3);
        *v19 = &unk_49DB368;
        v21 = v19[3];
        if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
          sub_BD60C0(v19 + 1);
        v19 += 9;
      }
      while ( (_QWORD *)v18 != v19 );
      v19 = (_QWORD *)v33[84];
    }
    if ( v19 )
      j_j___libc_free_0((unsigned __int64)v19);
    if ( !BYTE4(v33[74]) )
      _libc_free(v33[72]);
    if ( (unsigned __int64 *)v33[0] != &v33[2] )
      _libc_free(v33[0]);
    return v12;
  }
LABEL_23:
  v12 = 0;
  v23 = sub_B2DBE0((__int64)a2);
  v3 = "shadow-stack";
  if ( !sub_2241AC0(v23, "shadow-stack") )
  {
    v3 = (const char *)a2;
    v12 = sub_2FA5DB0(v11, a2, (__int64)v13);
  }
  if ( LOBYTE(v33[87]) )
    goto LABEL_6;
  return v12;
}
