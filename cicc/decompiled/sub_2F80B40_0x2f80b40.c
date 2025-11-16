// Function: sub_2F80B40
// Address: 0x2f80b40
//
__int64 __fastcall sub_2F80B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 **v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void **v10; // rax
  __int64 **v11; // rdx
  __int64 v12; // rcx
  void **v13; // rax
  int v14; // eax
  void **v15; // rax
  __int64 **v16; // rdx
  __int64 v17; // rcx
  __int64 **v18; // rax
  int v19; // eax
  __int64 **v20; // rax
  __int64 **v22; // rdi
  __int64 *v23; // rax
  __int64 **v24; // rdi
  __int64 **v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // [rsp+0h] [rbp-80h] BYREF
  unsigned __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  char v30; // [rsp+1Ch] [rbp-64h]
  char v31[16]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v32; // [rsp+30h] [rbp-50h] BYREF
  void **v33; // [rsp+38h] [rbp-48h]
  unsigned int v34; // [rsp+44h] [rbp-3Ch]
  int v35; // [rsp+48h] [rbp-38h]
  char v36; // [rsp+4Ch] [rbp-34h]
  char v37[48]; // [rsp+50h] [rbp-30h] BYREF

  v29 = 0;
  v27 = sub_2EB2140(a4, (__int64 *)&unk_501EAD0, a3) + 8;
  v28 = *(_QWORD *)(a3 + 32);
  if ( !*(_BYTE *)(v28 + 48) || !(unsigned __int8)sub_2F80AA0(&v27, *(_QWORD *)(a3 + 16)) )
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0((__int64)&v27);
  if ( v34 != v35 )
    goto LABEL_4;
  if ( v30 )
  {
    v10 = (void **)v28;
    v22 = (__int64 **)(v28 + 8LL * HIDWORD(v29));
    v7 = HIDWORD(v29);
    v6 = (__int64 **)v28;
    if ( (__int64 **)v28 == v22 )
      goto LABEL_67;
    while ( *v6 != &qword_4F82400 )
    {
      if ( v22 == ++v6 )
      {
LABEL_8:
        while ( *v10 != &unk_4F82408 )
        {
          if ( ++v10 == (void **)v6 )
            goto LABEL_67;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v27, (__int64)&qword_4F82400) )
  {
LABEL_4:
    if ( !v30 )
    {
LABEL_69:
      sub_C8CC70((__int64)&v27, (__int64)&unk_4F82408, (__int64)v6, v7, v8, v9);
      goto LABEL_9;
    }
    v10 = (void **)v28;
    v7 = HIDWORD(v29);
    v6 = (__int64 **)(v28 + 8LL * HIDWORD(v29));
    if ( v6 != (__int64 **)v28 )
      goto LABEL_8;
LABEL_67:
    if ( (unsigned int)v29 > (unsigned int)v7 )
    {
      HIDWORD(v29) = v7 + 1;
      *v6 = (__int64 *)&unk_4F82408;
      ++v27;
      goto LABEL_9;
    }
    goto LABEL_69;
  }
LABEL_9:
  if ( !v36 )
  {
    v26 = sub_C8CA60((__int64)&v32, (__int64)&unk_501EAD0);
    if ( v26 )
    {
      *v26 = -2;
      ++v32;
      v12 = v34;
      v14 = ++v35;
    }
    else
    {
      v12 = v34;
      v14 = v35;
    }
    goto LABEL_15;
  }
  v11 = (__int64 **)&v33[v34];
  v12 = v34;
  v13 = v33;
  if ( v33 != (void **)v11 )
  {
    while ( *v13 != &unk_501EAD0 )
    {
      if ( v11 == (__int64 **)++v13 )
        goto LABEL_56;
    }
    v11 = (__int64 **)v33[--v34];
    *v13 = v11;
    v12 = v34;
    ++v32;
    v14 = v35;
LABEL_15:
    if ( (_DWORD)v12 != v14 )
      goto LABEL_16;
    goto LABEL_57;
  }
LABEL_56:
  if ( v34 != v35 )
    goto LABEL_16;
LABEL_57:
  if ( v30 )
  {
    v15 = (void **)v28;
    v25 = (__int64 **)(v28 + 8LL * HIDWORD(v29));
    v12 = HIDWORD(v29);
    v11 = (__int64 **)v28;
    if ( (__int64 **)v28 != v25 )
    {
      while ( *v11 != &qword_4F82400 )
      {
        if ( v25 == ++v11 )
        {
LABEL_20:
          while ( *v15 != &unk_501EAD0 )
          {
            if ( ++v15 == (void **)v11 )
              goto LABEL_65;
          }
          break;
        }
      }
LABEL_21:
      if ( v36 )
        goto LABEL_22;
LABEL_46:
      v23 = sub_C8CA60((__int64)&v32, (__int64)&qword_5025C20);
      if ( v23 )
      {
        *v23 = -2;
        v17 = v34;
        ++v32;
        if ( v34 != ++v35 )
          goto LABEL_28;
        goto LABEL_48;
      }
      v17 = v34;
      goto LABEL_55;
    }
    goto LABEL_65;
  }
  if ( sub_C8CA60((__int64)&v27, (__int64)&qword_4F82400) )
    goto LABEL_21;
LABEL_16:
  if ( !v30 )
    goto LABEL_45;
  v15 = (void **)v28;
  v12 = HIDWORD(v29);
  v11 = (__int64 **)(v28 + 8LL * HIDWORD(v29));
  if ( (__int64 **)v28 != v11 )
    goto LABEL_20;
LABEL_65:
  if ( (unsigned int)v12 < (unsigned int)v29 )
  {
    HIDWORD(v29) = v12 + 1;
    *v11 = (__int64 *)&unk_501EAD0;
    ++v27;
    goto LABEL_21;
  }
LABEL_45:
  sub_C8CC70((__int64)&v27, (__int64)&unk_501EAD0, (__int64)v11, v12, v8, v9);
  if ( !v36 )
    goto LABEL_46;
LABEL_22:
  v16 = (__int64 **)&v33[v34];
  v17 = v34;
  if ( v33 == (void **)v16 )
  {
LABEL_55:
    v19 = v35;
    goto LABEL_27;
  }
  v18 = (__int64 **)v33;
  while ( *v18 != &qword_5025C20 )
  {
    if ( v16 == ++v18 )
      goto LABEL_55;
  }
  v16 = (__int64 **)v33[--v34];
  *v18 = (__int64 *)v16;
  v17 = v34;
  ++v32;
  v19 = v35;
LABEL_27:
  if ( (_DWORD)v17 != v19 )
  {
LABEL_28:
    if ( !v30 )
    {
LABEL_72:
      sub_C8CC70((__int64)&v27, (__int64)&qword_5025C20, (__int64)v16, v17, v8, v9);
      goto LABEL_33;
    }
    v20 = (__int64 **)v28;
    v17 = HIDWORD(v29);
    v16 = (__int64 **)(v28 + 8LL * HIDWORD(v29));
    if ( (__int64 **)v28 != v16 )
      goto LABEL_32;
LABEL_70:
    if ( (unsigned int)v17 < (unsigned int)v29 )
    {
      HIDWORD(v29) = v17 + 1;
      *v16 = &qword_5025C20;
      ++v27;
      goto LABEL_33;
    }
    goto LABEL_72;
  }
LABEL_48:
  if ( v30 )
  {
    v20 = (__int64 **)v28;
    v24 = (__int64 **)(v28 + 8LL * HIDWORD(v29));
    v17 = HIDWORD(v29);
    v16 = (__int64 **)v28;
    if ( (__int64 **)v28 == v24 )
      goto LABEL_70;
    while ( *v16 != &qword_4F82400 )
    {
      if ( v24 == ++v16 )
      {
LABEL_32:
        while ( *v20 != &qword_5025C20 )
        {
          if ( ++v20 == v16 )
            goto LABEL_70;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v27, (__int64)&qword_4F82400) )
  {
    goto LABEL_28;
  }
LABEL_33:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v31, (__int64)&v27);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v37, (__int64)&v32);
  if ( !v36 )
    _libc_free((unsigned __int64)v33);
  if ( !v30 )
  {
    _libc_free(v28);
    return a1;
  }
  return a1;
}
