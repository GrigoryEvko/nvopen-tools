// Function: sub_227D140
// Address: 0x227d140
//
void __fastcall sub_227D140(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, void *a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r10d
  unsigned int i; // eax
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rdi
  _QWORD *v17; // rax
  void **v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  void **v22; // rax
  void **v23; // rdx
  __int64 v24; // rcx
  void **v25; // rax
  int v26; // eax
  void **v27; // rax
  __int64 *v28; // rax
  unsigned __int64 v29; // [rsp+0h] [rbp-B0h]
  __int64 *v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+20h] [rbp-90h] BYREF
  void **v32; // [rsp+28h] [rbp-88h]
  __int64 v33; // [rsp+30h] [rbp-80h]
  int v34; // [rsp+38h] [rbp-78h]
  char v35; // [rsp+3Ch] [rbp-74h]
  char v36; // [rsp+40h] [rbp-70h] BYREF
  __int64 v37; // [rsp+50h] [rbp-60h] BYREF
  void **v38; // [rsp+58h] [rbp-58h]
  __int64 v39; // [rsp+60h] [rbp-50h]
  int v40; // [rsp+68h] [rbp-48h]
  char v41; // [rsp+6Ch] [rbp-44h]
  char v42; // [rsp+70h] [rbp-40h] BYREF

  v30 = &a2[a3];
  if ( v30 != a2 )
  {
    v7 = a2;
    while ( 1 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = *v7;
      v10 = *(unsigned int *)(v8 + 88);
      v11 = *(_QWORD *)(v8 + 72);
      if ( !(_DWORD)v10 )
        goto LABEL_56;
      v12 = 1;
      v29 = (unsigned __int64)(((unsigned int)&unk_4FDADA8 >> 9) ^ ((unsigned int)&unk_4FDADA8 >> 4)) << 32;
      for ( i = (v10 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v29 | ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))) >> 31)
               ^ (484763065 * (v29 | ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)))); ; i = (v10 - 1) & v15 )
      {
        v14 = v11 + 24LL * i;
        a5 = *(void **)v14;
        if ( *(_UNKNOWN **)v14 == &unk_4FDADA8 && v9 == *(_QWORD *)(v14 + 8) )
          break;
        if ( a5 == (void *)-4096LL && *(_QWORD *)(v14 + 8) == -4096 )
          goto LABEL_56;
        v15 = v12 + i;
        ++v12;
      }
      if ( v14 == v11 + 24 * v10 )
LABEL_56:
        v10 = 0;
      else
        LOBYTE(v10) = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL) != 0;
      **(_BYTE **)a1 |= v10;
      v16 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL);
      if ( !*(_BYTE *)(v16 + 28) )
        break;
      v17 = *(_QWORD **)(v16 + 8);
      v11 = *(unsigned int *)(v16 + 20);
      v10 = (__int64)&v17[v11];
      if ( v17 == (_QWORD *)v10 )
      {
LABEL_46:
        if ( (unsigned int)v11 >= *(_DWORD *)(v16 + 16) )
          break;
        v11 = (unsigned int)(v11 + 1);
        *(_DWORD *)(v16 + 20) = v11;
        *(_QWORD *)v10 = v9;
        ++*(_QWORD *)v16;
      }
      else
      {
        while ( v9 != *v17 )
        {
          if ( (_QWORD *)v10 == ++v17 )
            goto LABEL_46;
        }
      }
LABEL_16:
      v31 = 0;
      v33 = 2;
      v32 = (void **)&v36;
      v34 = 0;
      v35 = 1;
      v37 = 0;
      v38 = (void **)&v42;
      v39 = 2;
      v40 = 0;
      v41 = 1;
      if ( (unsigned __int8)sub_B19060((__int64)&v31, (__int64)&unk_4F82400, v10, v11) )
        goto LABEL_22;
      if ( !v35 )
        goto LABEL_54;
      v22 = v32;
      v19 = HIDWORD(v33);
      v18 = &v32[HIDWORD(v33)];
      if ( v32 == v18 )
      {
LABEL_52:
        if ( HIDWORD(v33) < (unsigned int)v33 )
        {
          ++HIDWORD(v33);
          *v18 = &unk_4F82420;
          ++v31;
          goto LABEL_22;
        }
LABEL_54:
        sub_C8CC70((__int64)&v31, (__int64)&unk_4F82420, (__int64)v18, v19, v20, v21);
        goto LABEL_22;
      }
      while ( *v22 != &unk_4F82420 )
      {
        if ( v18 == ++v22 )
          goto LABEL_52;
      }
LABEL_22:
      if ( v41 )
      {
        v23 = &v38[HIDWORD(v39)];
        v24 = HIDWORD(v39);
        v25 = v38;
        if ( v38 != v23 )
        {
          while ( *v25 != &unk_4FDADA8 )
          {
            if ( v23 == ++v25 )
              goto LABEL_43;
          }
          --HIDWORD(v39);
          v23 = (void **)v38[HIDWORD(v39)];
          *v25 = v23;
          v24 = HIDWORD(v39);
          ++v37;
          v26 = v40;
LABEL_28:
          if ( (_DWORD)v24 == v26 )
            goto LABEL_44;
          goto LABEL_29;
        }
      }
      else
      {
        v28 = sub_C8CA60((__int64)&v37, (__int64)&unk_4FDADA8);
        if ( v28 )
        {
          *v28 = -2;
          ++v37;
          v24 = HIDWORD(v39);
          v26 = ++v40;
          goto LABEL_28;
        }
        v24 = HIDWORD(v39);
      }
LABEL_43:
      if ( (_DWORD)v24 == v40 )
      {
LABEL_44:
        if ( (unsigned __int8)sub_B19060((__int64)&v31, (__int64)&unk_4F82400, (__int64)v23, v24) )
          goto LABEL_34;
      }
LABEL_29:
      if ( !v35 )
        goto LABEL_41;
      v27 = v32;
      v24 = HIDWORD(v33);
      v23 = &v32[HIDWORD(v33)];
      if ( v32 != v23 )
      {
        while ( *v27 != &unk_4FDADA8 )
        {
          if ( v23 == ++v27 )
            goto LABEL_50;
        }
        goto LABEL_34;
      }
LABEL_50:
      if ( HIDWORD(v33) < (unsigned int)v33 )
      {
        v24 = (unsigned int)++HIDWORD(v33);
        *v23 = &unk_4FDADA8;
        ++v31;
      }
      else
      {
LABEL_41:
        sub_C8CC70((__int64)&v31, (__int64)&unk_4FDADA8, (__int64)v23, v24, v20, v21);
      }
LABEL_34:
      sub_227C930(*(_QWORD *)(a1 + 8), v9, (__int64)&v31, v24);
      if ( !v41 )
        _libc_free((unsigned __int64)v38);
      if ( !v35 )
        _libc_free((unsigned __int64)v32);
      if ( v30 == ++v7 )
        return;
    }
    sub_C8CC70(v16, v9, v10, v11, (__int64)a5, a6);
    goto LABEL_16;
  }
}
