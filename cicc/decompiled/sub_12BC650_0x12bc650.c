// Function: sub_12BC650
// Address: 0x12bc650
//
__int64 __fastcall sub_12BC650(__int64 *a1, __int64 (*a2)(), size_t a3, const char *a4, __int64 a5, __int64 a6)
{
  const char *v6; // r13
  char v8; // r14
  unsigned int v9; // r12d
  size_t v10; // r12
  __int64 v11; // rdi
  void *v12; // rax
  __int64 v13; // r15
  __int64 *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  void *v19; // rax
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 (*src)(); // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  size_t v25; // [rsp+28h] [rbp-48h]
  void *v26; // [rsp+30h] [rbp-40h]
  size_t v27; // [rsp+38h] [rbp-38h]

  v6 = a4;
  v8 = byte_4F92D70;
  src = a2;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
    {
      a2 = sub_12B9A60;
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    }
    v21 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 )
    {
      v9 = 5;
      goto LABEL_25;
    }
    if ( !src )
    {
      v9 = 4;
      goto LABEL_25;
    }
    v8 = 1;
LABEL_13:
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    if ( v6 )
    {
      v10 = strlen(v6);
      v11 = v10 + 1;
    }
    else
    {
      v11 = 10;
      v10 = 9;
      v6 = "<unnamed>";
    }
    v25 = v10;
    v12 = (void *)malloc(v11, a2, a3, a4, a5, a6);
    v24 = (__int64)v12;
    v13 = (__int64)v12;
    if ( v12 )
    {
      v14 = (__int64 *)v6;
      memcpy(v12, v6, v10);
      *(_BYTE *)(v13 + v10) = 0;
      v27 = a3;
      if ( a3 == -1 || (v19 = (void *)malloc(a3 + 1, v6, v15, v16, v17, v18), (v26 = v19) == 0) )
      {
        v9 = 1;
      }
      else
      {
        v14 = &v24;
        *((_BYTE *)memcpy(v19, src, a3) + a3) = 0;
        sub_12BC5E0(a1, &v24);
        if ( v26 )
          _libc_free(v26, &v24);
        v13 = v24;
        v9 = 0;
        if ( !v24 )
        {
LABEL_21:
          if ( !v8 )
            return v9;
LABEL_25:
          sub_16C30E0(v21);
          return v9;
        }
      }
      _libc_free(v13, v14);
      if ( !v8 )
        return v9;
      goto LABEL_25;
    }
    v9 = 1;
    goto LABEL_21;
  }
  if ( !qword_4F92D80 )
  {
    a2 = sub_12B9A60;
    sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
  }
  v21 = qword_4F92D80;
  if ( !a1 )
    return 5;
  if ( src )
    goto LABEL_13;
  return 4;
}
