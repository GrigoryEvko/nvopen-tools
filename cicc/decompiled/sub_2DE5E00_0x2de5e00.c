// Function: sub_2DE5E00
// Address: 0x2de5e00
//
__int64 __fastcall sub_2DE5E00(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 v8; // r13
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // r14d
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _QWORD *v27; // r14
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // r13
  _QWORD *v35; // r12
  void (__fastcall *v36)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v37; // rax
  unsigned __int64 v38; // [rsp+0h] [rbp-300h]
  unsigned __int64 v39; // [rsp+8h] [rbp-2F8h]
  unsigned __int64 v40[94]; // [rsp+10h] [rbp-2F0h] BYREF

  v4 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190);
  if ( !v4 )
    return 0;
  v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_5027190);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(v5 + 256);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 16LL);
  if ( v7 == sub_23CE270 )
    BUG();
  v8 = ((__int64 (__fastcall *)(__int64, __int64))v7)(v6, a2);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v8 + 320LL))(v8) )
  {
    v9 = *(__int64 (**)())(*(_QWORD *)v8 + 144LL);
    if ( v9 != sub_2C8F680 )
      ((void (__fastcall *)(__int64))v9)(v8);
    memset(v40, 0, 0x2C0u);
    v10 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
    if ( v10 && (v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v10 + 104LL))(v10, &unk_4F8144C)) != 0 )
    {
      v38 = v11 + 176;
      if ( LOBYTE(v40[87]) )
      {
        LOBYTE(v40[87]) = 0;
        sub_FFCE90((__int64)v40, (__int64)&unk_4F8144C, v12, v13, v14, v15);
        sub_FFD870((__int64)v40, (__int64)&unk_4F8144C, v23, v24, v25, v26);
        sub_FFBC40((__int64)v40, (__int64)&unk_4F8144C);
        v27 = (_QWORD *)v40[84];
        v39 = v40[85];
        if ( v40[85] != v40[84] )
        {
          do
          {
            v28 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v27[7];
            *v27 = &unk_49E5048;
            if ( v28 )
              v28(v27 + 5, v27 + 5, 3);
            *v27 = &unk_49DB368;
            v29 = v27[3];
            if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
              sub_BD60C0(v27 + 1);
            v27 += 9;
          }
          while ( (_QWORD *)v39 != v27 );
          v27 = (_QWORD *)v40[84];
        }
        if ( v27 )
          j_j___libc_free_0((unsigned __int64)v27);
        if ( !BYTE4(v40[74]) )
          _libc_free(v40[72]);
        if ( (unsigned __int64 *)v40[0] != &v40[2] )
          _libc_free(v40[0]);
      }
      LOBYTE(v40[87]) = 1;
      v16 = v40;
      v40[1] = 0x1000000000LL;
      v40[0] = (unsigned __int64)&v40[2];
      v40[68] = v38;
      v40[72] = (unsigned __int64)&v40[75];
      v40[66] = 0;
      v40[67] = 0;
      v40[69] = 0;
      LOBYTE(v40[70]) = 1;
      v40[71] = 0;
      v40[73] = 8;
      LODWORD(v40[74]) = 0;
      BYTE4(v40[74]) = 1;
      LOWORD(v40[83]) = 0;
      memset(&v40[84], 0, 24);
    }
    else
    {
      v16 = 0;
      if ( LOBYTE(v40[87]) )
        v16 = v40;
    }
    v21 = sub_2DE4C60(a2, (__int64)v16);
    if ( LOBYTE(v40[87]) )
    {
      LOBYTE(v40[87]) = 0;
      sub_FFCE90((__int64)v40, (__int64)v16, v17, v18, v19, v20);
      sub_FFD870((__int64)v40, (__int64)v16, v30, v31, v32, v33);
      sub_FFBC40((__int64)v40, (__int64)v16);
      v34 = v40[85];
      v35 = (_QWORD *)v40[84];
      if ( v40[85] != v40[84] )
      {
        do
        {
          v36 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v35[7];
          *v35 = &unk_49E5048;
          if ( v36 )
            v36(v35 + 5, v35 + 5, 3);
          *v35 = &unk_49DB368;
          v37 = v35[3];
          if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
            sub_BD60C0(v35 + 1);
          v35 += 9;
        }
        while ( (_QWORD *)v34 != v35 );
        v35 = (_QWORD *)v40[84];
      }
      if ( v35 )
        j_j___libc_free_0((unsigned __int64)v35);
      if ( !BYTE4(v40[74]) )
        _libc_free(v40[72]);
      if ( (unsigned __int64 *)v40[0] != &v40[2] )
        _libc_free(v40[0]);
    }
  }
  else
  {
    return 0;
  }
  return v21;
}
