// Function: sub_1C1F7B0
// Address: 0x1c1f7b0
//
void __fastcall sub_1C1F7B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  _QWORD *v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rcx
  char *v12; // rax
  char *v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  char v16; // al
  _BOOL8 v17; // rcx
  __int64 v18; // rdx
  char *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rbx
  void *v22; // rax
  char *v23; // rdx
  char v24; // [rsp+7h] [rbp-69h] BYREF
  __int64 v25; // [rsp+8h] [rbp-68h] BYREF
  void *src; // [rsp+10h] [rbp-60h] BYREF
  char *v27; // [rsp+18h] [rbp-58h]
  char *v28; // [rsp+20h] [rbp-50h]
  char *v29; // [rsp+30h] [rbp-40h] BYREF
  __int64 v30; // [rsp+38h] [rbp-38h]
  __int64 v31; // [rsp+40h] [rbp-30h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v4 = *(_QWORD *)(a3 + 8);
    v5 = *(_QWORD **)a3;
    src = 0;
    v27 = 0;
    v6 = v4;
    v28 = 0;
    if ( (unsigned __int64)(8 * v4) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v7 = 0;
    if ( v6 * 8 )
    {
      v7 = (_QWORD *)sub_22077B0(8 * v4);
      v8 = &v7[v6];
      src = v7;
      v28 = (char *)&v7[v6];
      do
      {
        if ( v7 )
          *v7 = *v5;
        ++v7;
        ++v5;
      }
      while ( v8 != v7 );
    }
    v27 = (char *)v7;
    v9 = *(_QWORD *)a1;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v10 = (*(__int64 (__fastcall **)(__int64))(v9 + 16))(a1);
    v11 = 0;
    if ( v10 )
    {
      v12 = (char *)src;
      v13 = v29;
      if ( v27 - (_BYTE *)src == v30 - (_QWORD)v29 )
      {
        if ( src == v27 )
        {
LABEL_38:
          v11 = 1;
        }
        else
        {
          while ( *((_DWORD *)v12 + 1) == *((_DWORD *)v13 + 1)
               && ((*(_DWORD *)v13 ^ *(_DWORD *)v12) & 0xFFFFFF) == 0
               && v12[3] == v13[3] )
          {
            v12 += 8;
            v13 += 8;
            if ( v27 == v12 )
              goto LABEL_38;
          }
          v11 = 0;
        }
      }
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v11,
           &v24,
           &v25) )
    {
      sub_1C1F4D0(a1, (__int64)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
    }
    else if ( v24 )
    {
      sub_1C13B70((__int64)&src, &v29, v14);
    }
    if ( v29 )
      j_j___libc_free_0(v29, v31 - (_QWORD)v29);
    if ( src )
      j_j___libc_free_0(src, v28 - (_BYTE *)src);
  }
  else
  {
    v15 = *(_QWORD *)a1;
    src = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v16 = (*(__int64 (__fastcall **)(__int64))(v15 + 16))(a1);
    v17 = 0;
    if ( v16 )
      v17 = v27 - (_BYTE *)src == v30 - (_QWORD)v29;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           a2,
           0,
           v17,
           &v24,
           &v25) )
    {
      sub_1C1F4D0(a1, (__int64)&src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
    }
    else if ( v24 )
    {
      sub_1C13B70((__int64)&src, &v29, v18);
    }
    if ( v29 )
      j_j___libc_free_0(v29, v31 - (_QWORD)v29);
    v19 = (char *)src;
    if ( src == v27 )
    {
      *(_QWORD *)a3 = 0;
      *(_QWORD *)(a3 + 8) = 0;
    }
    else
    {
      v20 = sub_16E4080(a1);
      v21 = (v27 - (_BYTE *)src) >> 3;
      v22 = (void *)sub_145CBF0(*(__int64 **)(v20 + 8), v27 - (_BYTE *)src, 4);
      v23 = v27;
      v19 = (char *)src;
      *(_QWORD *)(a3 + 8) = v21;
      *(_QWORD *)a3 = v22;
      if ( v19 != v23 )
        memmove(v22, v19, v23 - v19);
    }
    if ( v19 )
      j_j___libc_free_0(v19, v28 - v19);
  }
}
