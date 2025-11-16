// Function: sub_186FC40
// Address: 0x186fc40
//
_QWORD *sub_186FC40()
{
  __int64 v0; // rax
  _QWORD *v1; // r12
  size_t v2; // rdx
  __int64 v3; // rbx
  void *v4; // r13
  size_t v5; // r14
  unsigned int v6; // r8d
  _QWORD *v7; // r9
  __int64 v8; // rax
  unsigned int v9; // r8d
  __int64 *v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rax
  void *v14; // rax
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r8
  bool v19; // zf
  __int64 v20; // r13
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  __int64 v23; // [rsp+10h] [rbp-B0h]
  __int64 *v24; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+18h] [rbp-A8h]
  unsigned int v26; // [rsp+20h] [rbp-A0h]
  __int64 v27; // [rsp+20h] [rbp-A0h]
  __int64 *v28; // [rsp+20h] [rbp-A0h]
  __int64 *v29; // [rsp+28h] [rbp-98h]
  unsigned int v30; // [rsp+28h] [rbp-98h]
  unsigned int v31; // [rsp+30h] [rbp-90h]
  unsigned __int64 v32; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v33; // [rsp+48h] [rbp-78h]
  __int64 v34; // [rsp+50h] [rbp-70h]
  void *src; // [rsp+60h] [rbp-60h] BYREF
  size_t n; // [rsp+68h] [rbp-58h]
  _QWORD v37[2]; // [rsp+70h] [rbp-50h] BYREF
  char v38; // [rsp+80h] [rbp-40h]

  v0 = sub_22077B0(192);
  v1 = (_QWORD *)v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = qword_4FABFC8;
    *(_QWORD *)(v0 + 16) = &unk_4FABE2C;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F1908;
    *(_DWORD *)(v0 + 24) = 5;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0x1000000000LL;
    if ( v2 )
      sub_186F640((__int64)&v32, (char *)qword_4FABFC0, v2);
    v3 = qword_4FABEE0;
    v23 = qword_4FABEE8;
    if ( qword_4FABEE8 != qword_4FABEE0 )
    {
      while ( 1 )
      {
        src = v37;
        sub_186F070((__int64 *)&src, *(_BYTE **)v3, *(_QWORD *)v3 + *(_QWORD *)(v3 + 8));
        v4 = src;
        v5 = n;
        v38 = 0;
        v6 = sub_16D19C0((__int64)&v32, (unsigned __int8 *)src, n);
        v7 = (_QWORD *)(v32 + 8LL * v6);
        if ( !*v7 )
          goto LABEL_12;
        if ( *v7 == -8 )
          break;
LABEL_6:
        if ( src != v37 )
          j_j___libc_free_0(src, v37[0] + 1LL);
        v3 += 32;
        if ( v23 == v3 )
          goto LABEL_19;
      }
      LODWORD(v34) = v34 - 1;
LABEL_12:
      v24 = (__int64 *)(v32 + 8LL * v6);
      v26 = v6;
      v8 = malloc(v5 + 17);
      v9 = v26;
      v10 = v24;
      v11 = v8;
      if ( v8 )
      {
        v12 = v8 + 16;
      }
      else
      {
        if ( v5 == -17 )
        {
          v13 = malloc(1u);
          v9 = v26;
          v10 = v24;
          v11 = 0;
          if ( v13 )
          {
            v12 = v13 + 16;
            v11 = v13;
            goto LABEL_18;
          }
        }
        v25 = v11;
        v28 = v10;
        v30 = v9;
        sub_16BD1C0("Allocation failed", 1u);
        v9 = v30;
        v12 = 16;
        v10 = v28;
        v11 = v25;
      }
      if ( v5 + 1 <= 1 )
      {
LABEL_15:
        *(_BYTE *)(v12 + v5) = 0;
        *(_QWORD *)v11 = v5;
        *(_BYTE *)(v11 + 8) = 0;
        *v10 = v11;
        ++HIDWORD(v33);
        sub_16D1CD0((__int64)&v32, v9);
        goto LABEL_6;
      }
LABEL_18:
      v27 = v11;
      v29 = v10;
      v31 = v9;
      v14 = memcpy((void *)v12, v4, v5);
      v11 = v27;
      v10 = v29;
      v9 = v31;
      v12 = (__int64)v14;
      goto LABEL_15;
    }
LABEL_19:
    v1[22] = 0;
    v15 = (unsigned __int64 *)sub_22077B0(32);
    if ( v15 )
    {
      v16 = v32;
      v1[20] = v15;
      v17 = 0;
      *v15 = v16;
      v15[1] = v33;
      v15[2] = v34;
      v1[23] = sub_186F370;
      v1[22] = sub_186F120;
    }
    else
    {
      v19 = HIDWORD(v33) == 0;
      v17 = v32;
      v1[20] = 0;
      v1[23] = sub_186F370;
      v1[22] = sub_186F120;
      if ( !v19 && (_DWORD)v33 )
      {
        v20 = 8LL * (unsigned int)v33;
        v21 = 0;
        do
        {
          v22 = *(_QWORD *)(v17 + v21);
          if ( v22 && v22 != -8 )
          {
            _libc_free(v22);
            v17 = v32;
          }
          v21 += 8;
        }
        while ( v21 != v20 );
      }
    }
    _libc_free(v17);
  }
  return v1;
}
