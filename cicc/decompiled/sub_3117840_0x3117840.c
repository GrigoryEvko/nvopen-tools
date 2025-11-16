// Function: sub_3117840
// Address: 0x3117840
//
void __fastcall sub_3117840(__int64 a1, _BYTE *a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 *v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 *v9; // rsi
  void *v10; // rsi
  void *v11; // rax
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  char *v14; // rcx
  size_t v15; // r12
  void *v16; // rdi
  void (__fastcall *v17)(__int64, unsigned __int64 **); // rax
  unsigned __int64 *v18; // rdi
  char v20; // [rsp+17h] [rbp-99h] BYREF
  unsigned __int64 v21; // [rsp+18h] [rbp-98h] BYREF
  unsigned __int64 v22; // [rsp+20h] [rbp-90h] BYREF
  int v23; // [rsp+28h] [rbp-88h]
  void *src; // [rsp+30h] [rbp-80h]
  _BYTE *v25; // [rsp+38h] [rbp-78h]
  __int64 v26; // [rsp+40h] [rbp-70h]
  unsigned __int64 *v27; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v28; // [rsp+58h] [rbp-58h]
  unsigned __int64 v29; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v30; // [rsp+68h] [rbp-48h]
  char *v31; // [rsp+70h] [rbp-40h]
  char *v32; // [rsp+78h] [rbp-38h]

  src = 0;
  v25 = 0;
  v26 = 0;
  if ( a2 )
  {
    v21 = a3;
    v27 = &v29;
    if ( a3 > 0xF )
    {
      v27 = (unsigned __int64 *)sub_22409D0((__int64)&v27, &v21, 0);
      v18 = v27;
      v29 = v21;
    }
    else
    {
      if ( a3 == 1 )
      {
        v7 = &v29;
        LOBYTE(v29) = *a2;
        v8 = 1;
LABEL_5:
        v28 = v8;
        *((_BYTE *)v7 + v8) = 0;
        v9 = v27;
        goto LABEL_7;
      }
      if ( !a3 )
      {
        v8 = 0;
        v7 = &v29;
        goto LABEL_5;
      }
      v18 = &v29;
    }
    memcpy(v18, a2, a3);
    v8 = v21;
    v7 = v27;
    goto LABEL_5;
  }
  v28 = 0;
  v27 = &v29;
  v9 = &v29;
  LOBYTE(v29) = 0;
LABEL_7:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64 *, __int64, _QWORD, char *, unsigned __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         v9,
         1,
         0,
         &v20,
         &v21) )
  {
    sub_3117620(a1, (__int64)&v22);
    (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  if ( v27 != &v29 )
    j_j___libc_free_0((unsigned __int64)v27);
  if ( sub_C93C90((__int64)a2, a3, 0, (unsigned __int64 *)&v27) || v27 != (unsigned __int64 *)(unsigned int)v27 )
  {
    v17 = *(void (__fastcall **)(__int64, unsigned __int64 **))(*(_QWORD *)a1 + 248LL);
    v27 = (unsigned __int64 *)"Id not an integer";
    LOWORD(v31) = 259;
    v17(a1, &v27);
    v16 = src;
    if ( src )
LABEL_21:
      j_j___libc_free_0((unsigned __int64)v16);
  }
  else
  {
    v10 = src;
    v30 = 0;
    v28 = v22;
    v31 = 0;
    LODWORD(v29) = v23;
    v11 = v25;
    v32 = 0;
    v12 = v25 - (_BYTE *)src;
    if ( v25 == src )
    {
      v15 = 0;
      v14 = 0;
    }
    else
    {
      if ( v12 > 0x7FFFFFFFFFFFFFFCLL )
        sub_4261EA(a2, src, (unsigned int)v27);
      v13 = sub_22077B0(v25 - (_BYTE *)src);
      v10 = src;
      v14 = (char *)v13;
      v11 = v25;
      v15 = v25 - (_BYTE *)src;
    }
    v30 = (unsigned __int64)v14;
    v31 = v14;
    v32 = &v14[v12];
    if ( v10 != v11 )
      v14 = (char *)memmove(v14, v10, v15);
    v31 = &v14[v15];
    sub_3115660(a4, (__int64)&v27);
    if ( v30 )
      j_j___libc_free_0(v30);
    v16 = src;
    if ( src )
      goto LABEL_21;
  }
}
