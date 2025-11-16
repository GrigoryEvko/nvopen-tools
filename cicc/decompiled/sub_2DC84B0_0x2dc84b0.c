// Function: sub_2DC84B0
// Address: 0x2dc84b0
//
__int64 __fastcall sub_2DC84B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 **v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void **v10; // rax
  void **v11; // rdx
  __int64 v12; // rcx
  void **v13; // rax
  int v14; // eax
  void **v15; // rax
  __int64 **v16; // rdx
  __int64 v17; // rcx
  void **v18; // rax
  int v19; // eax
  void **v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 **v24; // rdi
  void **v25; // rdi
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 **v28; // rdi
  __int64 v29; // [rsp+0h] [rbp-90h] BYREF
  void **v30; // [rsp+8h] [rbp-88h]
  unsigned int v31; // [rsp+10h] [rbp-80h]
  unsigned int v32; // [rsp+14h] [rbp-7Ch]
  char v33; // [rsp+1Ch] [rbp-74h]
  __int64 v34; // [rsp+30h] [rbp-60h] BYREF
  void **v35; // [rsp+38h] [rbp-58h]
  unsigned int v36; // [rsp+44h] [rbp-4Ch]
  int v37; // [rsp+48h] [rbp-48h]
  char v38; // [rsp+4Ch] [rbp-44h]

  v4 = a1 + 32;
  v29 = 0;
  v30 = 0;
  if ( !(unsigned __int8)sub_2DC80E0(&v29, a3) )
  {
    *(_QWORD *)(a1 + 8) = v4;
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
  if ( v36 != v37 )
    goto LABEL_5;
  if ( v33 )
  {
    v10 = v30;
    v28 = (__int64 **)&v30[v32];
    v7 = v32;
    v6 = (__int64 **)v30;
    if ( v30 == (void **)v28 )
      goto LABEL_63;
    while ( *v6 != &qword_4F82400 )
    {
      if ( v28 == ++v6 )
      {
LABEL_9:
        while ( *v10 != &unk_4F82408 )
        {
          if ( ++v10 == (void **)v6 )
            goto LABEL_63;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v29, (__int64)&qword_4F82400) )
  {
LABEL_5:
    if ( !v33 )
    {
LABEL_65:
      sub_C8CC70((__int64)&v29, (__int64)&unk_4F82408, (__int64)v6, v7, v8, v9);
      goto LABEL_10;
    }
    v10 = v30;
    v7 = v32;
    v6 = (__int64 **)&v30[v32];
    if ( v6 != (__int64 **)v30 )
      goto LABEL_9;
LABEL_63:
    if ( v31 > (unsigned int)v7 )
    {
      v32 = v7 + 1;
      *v6 = (__int64 *)&unk_4F82408;
      ++v29;
      goto LABEL_10;
    }
    goto LABEL_65;
  }
LABEL_10:
  if ( v38 )
  {
    v11 = &v35[v36];
    v12 = v36;
    if ( v35 == v11 )
    {
LABEL_46:
      if ( v37 != v36 )
        goto LABEL_17;
      goto LABEL_47;
    }
    v13 = v35;
    while ( *v13 != &unk_50208B0 )
    {
      if ( v11 == ++v13 )
        goto LABEL_46;
    }
    v11 = (void **)v35[--v36];
    *v13 = v11;
    v12 = v36;
    ++v34;
    v14 = v37;
  }
  else
  {
    v26 = sub_C8CA60((__int64)&v34, (__int64)&unk_50208B0);
    if ( v26 )
    {
      *v26 = -2;
      ++v34;
      v12 = v36;
      v14 = ++v37;
    }
    else
    {
      v12 = v36;
      v14 = v37;
    }
  }
  if ( v14 != (_DWORD)v12 )
  {
LABEL_17:
    if ( !v33 )
    {
LABEL_62:
      sub_C8CC70((__int64)&v29, (__int64)&unk_50208B0, (__int64)v11, v12, v8, v9);
      goto LABEL_22;
    }
    v15 = v30;
    v12 = v32;
    v11 = &v30[v32];
    if ( v30 != v11 )
      goto LABEL_21;
LABEL_60:
    if ( (unsigned int)v12 < v31 )
    {
      v32 = v12 + 1;
      *v11 = &unk_50208B0;
      ++v29;
      goto LABEL_22;
    }
    goto LABEL_62;
  }
LABEL_47:
  if ( v33 )
  {
    v15 = v30;
    v25 = &v30[v32];
    v12 = v32;
    v11 = v30;
    if ( v30 == v25 )
      goto LABEL_60;
    while ( *v11 != &qword_4F82400 )
    {
      if ( v25 == ++v11 )
      {
LABEL_21:
        while ( *v15 != &unk_50208B0 )
        {
          if ( ++v15 == v11 )
            goto LABEL_60;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v29, (__int64)&qword_4F82400) )
  {
    goto LABEL_17;
  }
LABEL_22:
  if ( v38 )
  {
    v16 = (__int64 **)&v35[v36];
    v17 = v36;
    if ( v35 != (void **)v16 )
    {
      v18 = v35;
      while ( *v18 != &unk_501FE48 )
      {
        if ( v16 == (__int64 **)++v18 )
          goto LABEL_39;
      }
      v16 = (__int64 **)v35[--v36];
      *v18 = v16;
      v17 = v36;
      ++v34;
      v19 = v37;
LABEL_28:
      if ( (_DWORD)v17 != v19 )
        goto LABEL_29;
      goto LABEL_40;
    }
  }
  else
  {
    v27 = sub_C8CA60((__int64)&v34, (__int64)&unk_501FE48);
    if ( v27 )
    {
      *v27 = -2;
      ++v34;
      v17 = v36;
      v19 = ++v37;
      goto LABEL_28;
    }
    v17 = v36;
  }
LABEL_39:
  if ( (_DWORD)v17 != v37 )
    goto LABEL_29;
LABEL_40:
  if ( v33 )
  {
    v20 = v30;
    v24 = (__int64 **)&v30[v32];
    v17 = v32;
    v16 = (__int64 **)v30;
    if ( v30 != (void **)v24 )
    {
      while ( *v16 != &qword_4F82400 )
      {
        if ( v24 == ++v16 )
        {
LABEL_33:
          while ( *v20 != &unk_501FE48 )
          {
            if ( ++v20 == (void **)v16 )
              goto LABEL_57;
          }
          goto LABEL_34;
        }
      }
      goto LABEL_34;
    }
    goto LABEL_57;
  }
  if ( sub_C8CA60((__int64)&v29, (__int64)&qword_4F82400) )
    goto LABEL_34;
LABEL_29:
  if ( !v33 )
  {
LABEL_59:
    sub_C8CC70((__int64)&v29, (__int64)&unk_501FE48, (__int64)v16, v17, v8, v9);
    goto LABEL_34;
  }
  v20 = v30;
  v17 = v32;
  v16 = (__int64 **)&v30[v32];
  if ( v16 != (__int64 **)v30 )
    goto LABEL_33;
LABEL_57:
  if ( (unsigned int)v17 >= v31 )
    goto LABEL_59;
  v17 = (unsigned int)(v17 + 1);
  v32 = v17;
  *v16 = (__int64 *)&unk_501FE48;
  ++v29;
LABEL_34:
  sub_C8CD80(a1, v4, (__int64)&v29, v17, v8, v9);
  sub_C8CD80(a1 + 48, a1 + 80, (__int64)&v34, v21, v22, v23);
  if ( !v38 )
    _libc_free((unsigned __int64)v35);
  if ( !v33 )
    _libc_free((unsigned __int64)v30);
  return a1;
}
