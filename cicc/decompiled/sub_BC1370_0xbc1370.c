// Function: sub_BC1370
// Address: 0xbc1370
//
__int64 __fastcall sub_BC1370(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rsi
  char v20; // al
  __int64 v21; // rcx
  void **v22; // rax
  __int64 v23; // rcx
  void **v24; // rdx
  __int64 v25; // rax
  void **v26; // rsi
  __int64 v27; // rax
  void **v28; // rdx
  __int64 v29; // rcx
  void **v30; // rax
  __int64 v31; // rcx
  int v32; // eax
  void **v33; // rax
  __int64 v34; // rcx
  void **v35; // rdx
  _QWORD *v37; // rax
  int v38; // eax
  void **v39; // rsi
  void **v40; // rsi
  __int64 v41; // [rsp+10h] [rbp-130h]
  __int64 v43; // [rsp+28h] [rbp-118h]
  _QWORD *v45; // [rsp+38h] [rbp-108h]
  __int64 v46; // [rsp+48h] [rbp-F8h] BYREF
  _BYTE v47[8]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-E8h]
  char v49; // [rsp+6Ch] [rbp-D4h]
  _BYTE v50[8]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+88h] [rbp-B8h]
  char v52; // [rsp+9Ch] [rbp-A4h]
  _QWORD v53[18]; // [rsp+B0h] [rbp-90h] BYREF

  v41 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v6 = *(_QWORD *)(sub_BC0510(a4, &unk_4F8A320, a3) + 8);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  v46 = v6;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  v7 = *(_QWORD *)(a3 + 32);
  v43 = a3 + 24;
  if ( v7 == a3 + 24 )
    goto LABEL_67;
  do
  {
    v8 = v7 - 56;
    if ( !v7 )
      v8 = 0;
    if ( !sub_B2FC80(v8) && (unsigned __int8)sub_BBBC50(&v46, *a2, v8) )
    {
      (*(void (__fastcall **)(_BYTE *, __int64, __int64, __int64))(*(_QWORD *)*a2 + 16LL))(v47, *a2, v8, v41);
      if ( *((_BYTE *)a2 + 8) )
      {
        memset(v53, 0, 0x60u);
        v10 = 0;
        BYTE4(v53[3]) = 1;
        v53[1] = &v53[4];
        LODWORD(v53[2]) = 2;
        v53[7] = &v53[10];
        LODWORD(v53[8]) = 2;
        BYTE4(v53[9]) = 1;
      }
      else
      {
        sub_C8CD80(v53, &v53[4], v47);
        sub_C8CD80(&v53[6], &v53[10], v50);
      }
      sub_BBE020(v41, v8, (__int64)v53, v10);
      if ( !BYTE4(v53[9]) )
        _libc_free(v53[7], v8);
      if ( !BYTE4(v53[3]) )
        _libc_free(v53[1], v8);
      if ( v46 )
      {
        v13 = *(_QWORD **)(v46 + 432);
        v45 = &v13[4 * *(unsigned int *)(v46 + 440)];
        if ( v13 != v45 )
        {
          v14 = *a2;
          do
          {
            v53[0] = 0;
            v15 = (_QWORD *)sub_22077B0(16);
            if ( v15 )
            {
              v15[1] = v8;
              *v15 = &unk_49DB0A8;
            }
            v16 = v53[0];
            v53[0] = v15;
            if ( v16 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
            v17 = v13;
            v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 32LL))(v14);
            if ( (v13[3] & 2) == 0 )
              v17 = (_QWORD *)*v13;
            (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *, _BYTE *))(v13[3] & 0xFFFFFFFFFFFFFFF8LL))(
              v17,
              v19,
              v18,
              v53,
              v47);
            if ( v53[0] )
              (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v53[0] + 8LL))(v53[0]);
            v13 += 4;
          }
          while ( v45 != v13 );
        }
      }
      sub_BBADB0(a1, (__int64)v47, v11, v12);
      if ( !v52 )
        _libc_free(v51, v47);
      if ( !v49 )
        _libc_free(v48, v47);
    }
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v43 != v7 );
  v20 = *(_BYTE *)(a1 + 28);
  v21 = *(unsigned int *)(a1 + 68);
  if ( *(_DWORD *)(a1 + 72) != (_DWORD)v21 )
    goto LABEL_30;
  if ( v20 )
  {
LABEL_67:
    v22 = *(void ***)(a1 + 8);
    v40 = &v22[*(unsigned int *)(a1 + 20)];
    v23 = *(unsigned int *)(a1 + 20);
    v24 = v22;
    if ( v22 == v40 )
      goto LABEL_61;
    while ( *v24 != &unk_4F82400 )
    {
      if ( v40 == ++v24 )
      {
LABEL_34:
        while ( *v22 != &unk_4F82420 )
        {
          if ( v24 == ++v22 )
            goto LABEL_61;
        }
        break;
      }
    }
LABEL_35:
    v25 = a1;
    if ( *(_BYTE *)(a1 + 76) )
      goto LABEL_36;
    goto LABEL_50;
  }
  if ( sub_C8CA60(a1, &unk_4F82400, v9, v21) )
    goto LABEL_35;
  v20 = *(_BYTE *)(a1 + 28);
LABEL_30:
  if ( !v20 )
    goto LABEL_49;
  v22 = *(void ***)(a1 + 8);
  v23 = *(unsigned int *)(a1 + 20);
  v24 = &v22[v23];
  if ( v24 != v22 )
    goto LABEL_34;
LABEL_61:
  if ( *(_DWORD *)(a1 + 16) > (unsigned int)v23 )
  {
    v23 = (unsigned int)(v23 + 1);
    *(_DWORD *)(a1 + 20) = v23;
    *v24 = &unk_4F82420;
    ++*(_QWORD *)a1;
    goto LABEL_35;
  }
LABEL_49:
  sub_C8CC70(a1, &unk_4F82420);
  v25 = a1;
  if ( *(_BYTE *)(a1 + 76) )
  {
LABEL_36:
    v26 = *(void ***)(v25 + 56);
    v27 = *(unsigned int *)(v25 + 68);
    v28 = &v26[v27];
    v29 = v27;
    v30 = v26;
    if ( v26 == v28 )
    {
LABEL_60:
      v32 = *(_DWORD *)(a1 + 72);
    }
    else
    {
      while ( *v30 != &unk_4F82418 )
      {
        if ( v28 == ++v30 )
          goto LABEL_60;
      }
      v31 = (unsigned int)(v29 - 1);
      *(_DWORD *)(a1 + 68) = v31;
      v28 = (void **)v26[v31];
      *v30 = v28;
      v29 = *(unsigned int *)(a1 + 68);
      ++*(_QWORD *)(a1 + 48);
      v32 = *(_DWORD *)(a1 + 72);
    }
    goto LABEL_41;
  }
LABEL_50:
  v37 = (_QWORD *)sub_C8CA60(a1 + 48, &unk_4F82418, v24, v23);
  if ( v37 )
  {
    *v37 = -2;
    v38 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 48);
    *(_DWORD *)(a1 + 72) = ++v38;
    v29 = *(unsigned int *)(a1 + 68);
    if ( (_DWORD)v29 != v38 )
      goto LABEL_42;
    goto LABEL_52;
  }
  v29 = *(unsigned int *)(a1 + 68);
  v32 = *(_DWORD *)(a1 + 72);
LABEL_41:
  if ( (_DWORD)v29 != v32 )
  {
LABEL_42:
    if ( !*(_BYTE *)(a1 + 28) )
    {
LABEL_59:
      sub_C8CC70(a1, &unk_4F82418);
      return a1;
    }
    v33 = *(void ***)(a1 + 8);
    v34 = *(unsigned int *)(a1 + 20);
    v35 = &v33[v34];
    if ( v35 != v33 )
      goto LABEL_46;
LABEL_58:
    if ( (unsigned int)v34 < *(_DWORD *)(a1 + 16) )
    {
      *(_DWORD *)(a1 + 20) = v34 + 1;
      *v35 = &unk_4F82418;
      ++*(_QWORD *)a1;
      return a1;
    }
    goto LABEL_59;
  }
LABEL_52:
  if ( !*(_BYTE *)(a1 + 28) )
  {
    if ( sub_C8CA60(a1, &unk_4F82400, v28, v29) )
      return a1;
    goto LABEL_42;
  }
  v33 = *(void ***)(a1 + 8);
  v39 = &v33[*(unsigned int *)(a1 + 20)];
  LODWORD(v34) = *(_DWORD *)(a1 + 20);
  v35 = v33;
  if ( v33 == v39 )
    goto LABEL_58;
  while ( *v35 != &unk_4F82400 )
  {
    if ( v39 == ++v35 )
    {
LABEL_46:
      while ( *v33 != &unk_4F82418 )
      {
        if ( v35 == ++v33 )
          goto LABEL_58;
      }
      return a1;
    }
  }
  return a1;
}
