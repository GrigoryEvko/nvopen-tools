// Function: sub_2F309D0
// Address: 0x2f309d0
//
__int64 __fastcall sub_2F309D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 **v11; // rdx
  __int64 v12; // rcx
  __int64 **v13; // rax
  int v14; // eax
  __int64 *v15; // rax
  __int64 **v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 **v19; // rax
  int v20; // eax
  __int64 **v21; // rax
  unsigned int v22; // edi
  char v23; // al
  void **v24; // rax
  __int64 *v25; // rax
  __int64 **v26; // rdi
  _QWORD v27[7]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+48h] [rbp-A8h]
  __int64 v29; // [rsp+50h] [rbp-A0h]
  unsigned int v30; // [rsp+58h] [rbp-98h]
  __int64 v31; // [rsp+60h] [rbp-90h] BYREF
  __int64 **v32; // [rsp+68h] [rbp-88h]
  unsigned int v33; // [rsp+70h] [rbp-80h]
  unsigned int v34; // [rsp+74h] [rbp-7Ch]
  char v35; // [rsp+7Ch] [rbp-74h]
  char v36[16]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v37; // [rsp+90h] [rbp-60h] BYREF
  __int64 **v38; // [rsp+98h] [rbp-58h]
  unsigned int v39; // [rsp+A4h] [rbp-4Ch]
  int v40; // [rsp+A8h] [rbp-48h]
  char v41; // [rsp+ACh] [rbp-44h]
  char v42[64]; // [rsp+B0h] [rbp-40h] BYREF

  v5 = 0;
  if ( (_BYTE)qword_5022BE8 )
    v5 = sub_2EB2140(a4, qword_501FE48, a3) + 8;
  v27[0] = off_4A2A708;
  memset(&v27[1], 0, 24);
  v27[4] = v5;
  v27[5] = sub_2EB2140(a4, &qword_50208B0, a3) + 8;
  v27[6] = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  if ( (_BYTE)qword_5022B08 || !(unsigned __int8)sub_2F2D9F0(v27, a3) )
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
    goto LABEL_6;
  }
  sub_2EAFFB0((__int64)&v31);
  if ( v41 )
  {
    v11 = &v38[v39];
    v12 = v39;
    if ( v38 == v11 )
    {
LABEL_47:
      v14 = v40;
    }
    else
    {
      v13 = v38;
      while ( *v13 != qword_501FE48 )
      {
        if ( v11 == ++v13 )
          goto LABEL_47;
      }
      v11 = (__int64 **)v38[--v39];
      *v13 = (__int64 *)v11;
      v12 = v39;
      ++v37;
      v14 = v40;
    }
  }
  else
  {
    v15 = sub_C8CA60((__int64)&v37, (__int64)qword_501FE48);
    if ( v15 )
    {
      *v15 = -2;
      ++v37;
      v12 = v39;
      v14 = ++v40;
    }
    else
    {
      v12 = v39;
      v14 = v40;
    }
  }
  if ( (_DWORD)v12 != v14 )
    goto LABEL_16;
  if ( v35 )
  {
    v16 = v32;
    v26 = &v32[v34];
    v12 = v34;
    v11 = v32;
    if ( v32 == v26 )
      goto LABEL_50;
    while ( *v11 != &qword_4F82400 )
    {
      if ( v26 == ++v11 )
      {
LABEL_20:
        while ( *v16 != qword_501FE48 )
        {
          if ( ++v16 == v11 )
            goto LABEL_50;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v31, (__int64)&qword_4F82400) )
  {
LABEL_16:
    if ( !v35 )
    {
LABEL_52:
      sub_C8CC70((__int64)&v31, (__int64)qword_501FE48, (__int64)v11, v12, v9, v10);
      goto LABEL_21;
    }
    v16 = v32;
    v12 = v34;
    v11 = &v32[v34];
    if ( v11 != v32 )
      goto LABEL_20;
LABEL_50:
    if ( v33 > (unsigned int)v12 )
    {
      v34 = v12 + 1;
      *v11 = qword_501FE48;
      ++v31;
      goto LABEL_21;
    }
    goto LABEL_52;
  }
LABEL_21:
  if ( v41 )
  {
    v17 = (__int64)&v38[v39];
    v18 = v39;
    v19 = v38;
    if ( v38 == (__int64 **)v17 )
      goto LABEL_46;
    while ( *v19 != &qword_50208B0 )
    {
      if ( (__int64 **)v17 == ++v19 )
        goto LABEL_46;
    }
    v17 = (__int64)v38[--v39];
    *v19 = (__int64 *)v17;
    v18 = v39;
    ++v37;
    v20 = v40;
  }
  else
  {
    v25 = sub_C8CA60((__int64)&v37, (__int64)&qword_50208B0);
    if ( !v25 )
    {
      v18 = v39;
LABEL_46:
      v20 = v40;
      goto LABEL_27;
    }
    *v25 = -2;
    ++v37;
    v18 = v39;
    v20 = ++v40;
  }
LABEL_27:
  if ( v20 != (_DWORD)v18 )
  {
LABEL_28:
    if ( !v35 )
    {
LABEL_44:
      sub_C8CC70((__int64)&v31, (__int64)&qword_50208B0, v17, v18, v9, v10);
      goto LABEL_33;
    }
    v17 = v34;
    v21 = v32;
    v18 = (__int64)&v32[v34];
    v22 = v34;
    if ( v32 != (__int64 **)v18 )
      goto LABEL_32;
    goto LABEL_43;
  }
  if ( !v35 )
  {
    if ( sub_C8CA60((__int64)&v31, (__int64)&qword_4F82400) )
      goto LABEL_33;
    goto LABEL_28;
  }
  v17 = (__int64)v32;
  v9 = (__int64)&v32[v34];
  v22 = v34;
  v21 = v32;
  v18 = (__int64)v32;
  if ( v32 != (__int64 **)v9 )
  {
    while ( *(__int64 **)v18 != &qword_4F82400 )
    {
      v18 += 8;
      if ( v9 == v18 )
      {
LABEL_32:
        while ( *v21 != &qword_50208B0 )
        {
          if ( ++v21 == (__int64 **)v18 )
            goto LABEL_43;
        }
        goto LABEL_33;
      }
    }
    goto LABEL_62;
  }
LABEL_43:
  if ( v22 >= v33 )
    goto LABEL_44;
  v34 = v22 + 1;
  *(_QWORD *)v18 = &qword_50208B0;
  ++v31;
LABEL_33:
  v23 = v35;
  if ( v39 == v40 )
  {
    if ( !v35 )
    {
      if ( sub_C8CA60((__int64)&v31, (__int64)&qword_4F82400) )
        goto LABEL_39;
      v23 = v35;
      goto LABEL_34;
    }
    v17 = (__int64)v32;
    v22 = v34;
LABEL_62:
    v24 = (void **)v17;
    v18 = v17 + 8LL * v22;
    if ( v17 != v18 )
    {
      while ( *(__int64 **)v17 != &qword_4F82400 )
      {
        v17 += 8;
        if ( v18 == v17 )
        {
LABEL_38:
          while ( *v24 != &unk_4F82408 )
          {
            if ( ++v24 == (void **)v17 )
              goto LABEL_53;
          }
          goto LABEL_39;
        }
      }
      goto LABEL_39;
    }
    goto LABEL_53;
  }
LABEL_34:
  if ( !v23 )
  {
LABEL_55:
    sub_C8CC70((__int64)&v31, (__int64)&unk_4F82408, v17, v18, v9, v10);
    goto LABEL_39;
  }
  v24 = (void **)v32;
  v22 = v34;
  v17 = (__int64)&v32[v34];
  if ( v32 != (__int64 **)v17 )
    goto LABEL_38;
LABEL_53:
  if ( v22 >= v33 )
    goto LABEL_55;
  v34 = v22 + 1;
  *(_QWORD *)v17 = &unk_4F82408;
  ++v31;
LABEL_39:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v36, (__int64)&v31);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v42, (__int64)&v37);
  if ( !v41 )
    _libc_free((unsigned __int64)v38);
  if ( !v35 )
    _libc_free((unsigned __int64)v32);
LABEL_6:
  v27[0] = off_4A2A708;
  sub_C7D6A0(v28, 16LL * v30, 8);
  return a1;
}
