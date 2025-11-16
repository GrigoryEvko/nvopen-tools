// Function: sub_2FCE450
// Address: 0x2fce450
//
__int64 __fastcall sub_2FCE450(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // r12d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _QWORD *v27; // r15
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD *v35; // r15
  void (__fastcall *v36)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rdi
  __int64 v39; // [rsp+8h] [rbp-48h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  _QWORD *v41; // [rsp+18h] [rbp-38h]
  _QWORD *v42; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 184) = a2;
  *(_QWORD *)(a1 + 192) = *(_QWORD *)(a2 + 40);
  v3 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v3 )
  {
    v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F8144C);
    if ( v4 )
    {
      v39 = v4 + 176;
      v40 = a1 + 216;
      if ( *(_BYTE *)(a1 + 896) )
      {
        *(_BYTE *)(a1 + 896) = 0;
        sub_FFCE90(a1 + 200, (__int64)&unk_4F8144C, v5, v6, v7, v8);
        sub_FFD870(a1 + 200, (__int64)&unk_4F8144C, v23, v24, v25, v26);
        sub_FFBC40(a1 + 200, (__int64)&unk_4F8144C);
        v27 = *(_QWORD **)(a1 + 872);
        v41 = *(_QWORD **)(a1 + 880);
        if ( v41 != v27 )
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
          while ( v41 != v27 );
          v27 = *(_QWORD **)(a1 + 872);
        }
        if ( v27 )
          j_j___libc_free_0((unsigned __int64)v27);
        if ( !*(_BYTE *)(a1 + 796) )
          _libc_free(*(_QWORD *)(a1 + 776));
        v30 = *(_QWORD *)(a1 + 200);
        if ( v30 != v40 )
          _libc_free(v30);
      }
      *(_QWORD *)(a1 + 728) = 0;
      *(_QWORD *)(a1 + 736) = 0;
      *(_QWORD *)(a1 + 200) = v40;
      *(_QWORD *)(a1 + 208) = 0x1000000000LL;
      *(_QWORD *)(a1 + 752) = 0;
      *(_QWORD *)(a1 + 744) = v39;
      *(_BYTE *)(a1 + 760) = 1;
      *(_QWORD *)(a1 + 768) = 0;
      *(_QWORD *)(a1 + 776) = a1 + 800;
      *(_QWORD *)(a1 + 784) = 8;
      *(_DWORD *)(a1 + 792) = 0;
      *(_BYTE *)(a1 + 796) = 1;
      *(_WORD *)(a1 + 864) = 0;
      *(_QWORD *)(a1 + 872) = 0;
      *(_QWORD *)(a1 + 880) = 0;
      *(_QWORD *)(a1 + 888) = 0;
      *(_BYTE *)(a1 + 896) = 1;
    }
  }
  v9 = *(__int64 **)(a1 + 8);
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_45:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_5027190 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_45;
  }
  *(_QWORD *)(a1 + 176) = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(
                                        *(_QWORD *)(v10 + 8),
                                        &unk_5027190)
                                    + 256);
  *(_WORD *)(a1 + 941) = 0;
  v12 = sub_B2D810(a2, "stack-protector-buffer-size", 0x1Bu, 8);
  v13 = *(_QWORD *)(a1 + 184);
  *(_DWORD *)(a1 + 936) = v12;
  if ( (unsigned __int8)sub_2FCD340(v13, a1 + 904)
    && ((*(_BYTE *)(a2 + 2) & 8) == 0 || (v22 = sub_B2E500(a2), (unsigned int)sub_B2A630(v22) - 7 > 3)) )
  {
    v14 = a1 + 200;
    v15 = *(_QWORD *)(a1 + 184);
    if ( !*(_BYTE *)(a1 + 896) )
      v14 = 0;
    v20 = sub_2FC9E30(*(_QWORD *)(a1 + 176), v15, v14, (unsigned __int8 *)(a1 + 941), (_BYTE *)(a1 + 942));
    if ( *(_BYTE *)(a1 + 896) )
    {
      *(_BYTE *)(a1 + 896) = 0;
      sub_FFCE90(a1 + 200, v15, v16, v17, v18, v19);
      sub_FFD870(a1 + 200, v15, v31, v32, v33, v34);
      sub_FFBC40(a1 + 200, v15);
      v35 = *(_QWORD **)(a1 + 872);
      v42 = *(_QWORD **)(a1 + 880);
      if ( v42 != v35 )
      {
        do
        {
          v36 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v35[7];
          *v35 = &unk_49E5048;
          if ( v36 )
            v36(v35 + 5, v35 + 5, 3);
          *v35 = &unk_49DB368;
          v37 = v35[3];
          if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
            sub_BD60C0(v35 + 1);
          v35 += 9;
        }
        while ( v42 != v35 );
        v35 = *(_QWORD **)(a1 + 872);
      }
      if ( v35 )
        j_j___libc_free_0((unsigned __int64)v35);
      if ( !*(_BYTE *)(a1 + 796) )
        _libc_free(*(_QWORD *)(a1 + 776));
      v38 = *(_QWORD *)(a1 + 200);
      if ( v38 != a1 + 216 )
        _libc_free(v38);
    }
  }
  else
  {
    return 0;
  }
  return v20;
}
