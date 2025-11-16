// Function: sub_2E4E960
// Address: 0x2e4e960
//
__int64 __fastcall sub_2E4E960(__int64 a1, char *a2, __int64 a3)
{
  char v4; // al
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r13
  unsigned __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // r13
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  unsigned __int64 v17; // rdi
  __int64 **v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  void *v22; // r9
  void **v23; // rax
  __int64 **v24; // rsi
  __int64 *v25; // rax
  __int64 v26; // [rsp+0h] [rbp-140h]
  char v27; // [rsp+8h] [rbp-138h]
  void *v28; // [rsp+8h] [rbp-138h]
  __int64 v29; // [rsp+10h] [rbp-130h] BYREF
  void **v30; // [rsp+18h] [rbp-128h]
  __int64 v31; // [rsp+20h] [rbp-120h]
  char v32; // [rsp+28h] [rbp-118h]
  char v33; // [rsp+2Ch] [rbp-114h]
  __int64 v34; // [rsp+30h] [rbp-110h] BYREF
  __int64 v35; // [rsp+38h] [rbp-108h]
  __int64 v36; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int64 v37; // [rsp+48h] [rbp-F8h]
  _BYTE *v38; // [rsp+50h] [rbp-F0h]
  __int64 v39; // [rsp+58h] [rbp-E8h]
  _BYTE v40[64]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-A0h]
  __int64 v42; // [rsp+A8h] [rbp-98h]
  __int64 v43; // [rsp+B0h] [rbp-90h]
  unsigned int v44; // [rsp+B8h] [rbp-88h]
  __int64 v45; // [rsp+C0h] [rbp-80h]
  __int64 v46; // [rsp+C8h] [rbp-78h]
  __int64 v47; // [rsp+D0h] [rbp-70h]
  unsigned int v48; // [rsp+D8h] [rbp-68h]
  __int64 v49; // [rsp+E0h] [rbp-60h]
  _QWORD *v50; // [rsp+E8h] [rbp-58h]
  __int64 v51; // [rsp+F0h] [rbp-50h]
  unsigned int v52; // [rsp+F8h] [rbp-48h]
  char v53; // [rsp+100h] [rbp-40h]

  v4 = *a2;
  v29 = 0;
  v30 = 0;
  v53 = 0;
  if ( !v4 )
    v4 = qword_501F508;
  v31 = 0;
  v34 = 0;
  v32 = v4;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = v40;
  v39 = 0x800000000LL;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v27 = sub_2E4E590((__int64)&v29, a3);
  v5 = v52;
  if ( v52 )
  {
    v6 = v50;
    v7 = &v50[10 * v52];
    do
    {
      if ( *v6 != -4096 && *v6 != -8192 )
      {
        v8 = v6[1];
        if ( (_QWORD *)v8 != v6 + 3 )
          _libc_free(v8);
      }
      v6 += 10;
    }
    while ( v7 != v6 );
    v5 = v52;
  }
  sub_C7D6A0((__int64)v50, 80 * v5, 8);
  v9 = v48;
  if ( v48 )
  {
    v10 = v46;
    v11 = v46 + ((unsigned __int64)v48 << 7);
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
        {
          v12 = *(_QWORD *)(v10 + 88);
          if ( v12 != v10 + 104 )
            _libc_free(v12);
          if ( !*(_BYTE *)(v10 + 52) )
            break;
        }
        v10 += 128;
        if ( v11 == v10 )
          goto LABEL_19;
      }
      v13 = *(_QWORD *)(v10 + 32);
      v10 += 128;
      _libc_free(v13);
    }
    while ( v11 != v10 );
LABEL_19:
    v9 = v48;
  }
  sub_C7D6A0(v46, v9 << 7, 8);
  v14 = v44;
  if ( v44 )
  {
    v15 = v42;
    v16 = v42 + 56LL * v44;
    do
    {
      while ( *(_QWORD *)v15 == -8192 || *(_QWORD *)v15 == -4096 || *(_BYTE *)(v15 + 36) )
      {
        v15 += 56;
        if ( v16 == v15 )
          goto LABEL_27;
      }
      v17 = *(_QWORD *)(v15 + 16);
      v26 = v16;
      v15 += 56;
      _libc_free(v17);
      v16 = v26;
    }
    while ( v26 != v15 );
LABEL_27:
    v14 = v44;
  }
  sub_C7D6A0(v42, 56 * v14, 8);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  sub_C7D6A0(v35, 8LL * (unsigned int)v37, 8);
  if ( !v27 )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0(&v29);
  v22 = (void *)(a1 + 32);
  if ( HIDWORD(v38) == (_DWORD)v39 )
  {
    if ( v33 )
    {
      v23 = v30;
      v24 = (__int64 **)&v30[HIDWORD(v31)];
      v20 = HIDWORD(v31);
      v19 = (__int64 **)v30;
      if ( v30 != (void **)v24 )
      {
        while ( *v19 != &qword_4F82400 )
        {
          if ( v24 == ++v19 )
          {
LABEL_38:
            while ( *v23 != &unk_4F82408 )
            {
              if ( ++v23 == (void **)v19 )
                goto LABEL_49;
            }
            goto LABEL_39;
          }
        }
        goto LABEL_39;
      }
      goto LABEL_49;
    }
    v25 = sub_C8CA60((__int64)&v29, (__int64)&qword_4F82400);
    v22 = (void *)(a1 + 32);
    if ( v25 )
      goto LABEL_39;
  }
  if ( !v33 )
  {
LABEL_51:
    v28 = v22;
    sub_C8CC70((__int64)&v29, (__int64)&unk_4F82408, (__int64)v19, v20, v21, (__int64)v22);
    v22 = v28;
    goto LABEL_39;
  }
  v23 = v30;
  v20 = HIDWORD(v31);
  v19 = (__int64 **)&v30[HIDWORD(v31)];
  if ( v30 != (void **)v19 )
    goto LABEL_38;
LABEL_49:
  if ( (unsigned int)v31 <= (unsigned int)v20 )
    goto LABEL_51;
  HIDWORD(v31) = v20 + 1;
  *v19 = (__int64 *)&unk_4F82408;
  ++v29;
LABEL_39:
  sub_C8CF70(a1, v22, 2, (__int64)&v34, (__int64)&v29);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v40, (__int64)&v36);
  if ( !BYTE4(v39) )
    _libc_free(v37);
  if ( !v33 )
    _libc_free((unsigned __int64)v30);
  return a1;
}
