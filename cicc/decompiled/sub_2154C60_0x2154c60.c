// Function: sub_2154C60
// Address: 0x2154c60
//
void *__fastcall sub_2154C60(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  unsigned int v5; // ebx
  char v6; // r15
  __int64 v7; // r12
  __int64 v8; // rdx
  void *v9; // rdi
  signed __int64 v10; // r15
  char *v11; // rax
  char *v12; // rcx
  signed __int64 v13; // r15
  char *v14; // rax
  char *v15; // rcx
  signed __int64 v16; // r15
  void *result; // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // r14
  __int64 v21; // rdi
  int v22; // [rsp+2Ch] [rbp-E4h]
  void *v23; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE *v24; // [rsp+38h] [rbp-D8h]
  __int64 v25; // [rsp+40h] [rbp-D0h]
  void *src; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE *v27; // [rsp+58h] [rbp-B8h]
  __int64 v28; // [rsp+60h] [rbp-B0h]
  const char *v29; // [rsp+70h] [rbp-A0h] BYREF
  char *v30; // [rsp+78h] [rbp-98h]
  char *v31; // [rsp+80h] [rbp-90h]
  _QWORD v32[2]; // [rsp+90h] [rbp-80h] BYREF
  _QWORD v33[2]; // [rsp+A0h] [rbp-70h] BYREF
  void *v34; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v35; // [rsp+B8h] [rbp-58h]
  __int64 v36; // [rsp+C0h] [rbp-50h]
  __int64 v37; // [rsp+C8h] [rbp-48h]
  int v38; // [rsp+D0h] [rbp-40h]
  const char *v39; // [rsp+D8h] [rbp-38h]

  v39 = (const char *)v32;
  v29 = "opencl.kernels";
  v32[0] = v33;
  v32[1] = 0;
  LOBYTE(v33[0]) = 0;
  v38 = 1;
  v37 = 0;
  v36 = 0;
  v35 = 0;
  v34 = &unk_49EFBE0;
  LOWORD(v31) = 259;
  v3 = sub_1632310(a2, (__int64)&v29);
  if ( v3 )
  {
    v4 = v3;
    v22 = sub_161F520(v3);
    if ( v22 )
    {
      v5 = 0;
      v6 = 0;
      do
      {
        v7 = sub_161F530(v4, v5);
        sub_2154A40(&v23, a1, v7);
        if ( v24 - (_BYTE *)v23 == 40 )
        {
          v29 = 0;
          v30 = 0;
          v31 = 0;
          v14 = (char *)sub_22077B0(40);
          v29 = v14;
          v15 = v14;
          v30 = v14;
          v31 = v14 + 40;
          v16 = v24 - (_BYTE *)v23;
          if ( v24 != v23 )
            v15 = (char *)memmove(v14, v23, v24 - (_BYTE *)v23);
          v30 = &v15[v16];
          sub_214E460(a1, (__int64)&v34, v7, (__int64 *)&v29);
          if ( v29 )
            j_j___libc_free_0(v29, v31 - v29);
          v6 = 1;
        }
        sub_2154880(&src, a1, v7);
        v9 = src;
        if ( v27 != src )
        {
          v29 = 0;
          v30 = 0;
          v10 = v27 - (_BYTE *)src;
          v31 = 0;
          if ( (unsigned __int64)(v27 - (_BYTE *)src) > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(src, a1, v8);
          v11 = (char *)sub_22077B0(v10);
          v29 = v11;
          v12 = v11;
          v30 = v11;
          v31 = &v11[v10];
          v13 = v27 - (_BYTE *)src;
          if ( v27 != src )
            v12 = (char *)memmove(v11, src, v27 - (_BYTE *)src);
          v30 = &v12[v13];
          sub_214EBB0(a1, (__int64)&v34, v7, &v29);
          if ( v29 )
            j_j___libc_free_0(v29, v31 - v29);
          v9 = src;
          v6 = 1;
        }
        if ( v9 )
          j_j___libc_free_0(v9, v28 - (_QWORD)v9);
        if ( v23 )
          j_j___libc_free_0(v23, v25 - (_QWORD)v23);
        ++v5;
      }
      while ( v22 != v5 );
      if ( v6 )
      {
        v19 = *(_QWORD *)(a1 + 256);
        v29 = ".metadata_section {\n\n";
        LOWORD(v31) = 259;
        sub_38DD5A0(v19, &v29);
        v20 = *(_QWORD *)(a1 + 256);
        if ( v37 != v35 )
          sub_16E7BA0((__int64 *)&v34);
        LOWORD(v31) = 260;
        v29 = v39;
        sub_38DD5A0(v20, &v29);
        v21 = *(_QWORD *)(a1 + 256);
        v29 = "} // end of .metadata_section\n";
        LOWORD(v31) = 259;
        sub_38DD5A0(v21, &v29);
      }
    }
    result = sub_16E7BC0((__int64 *)&v34);
    v18 = v32[0];
    if ( (_QWORD *)v32[0] != v33 )
      return (void *)j_j___libc_free_0(v18, v33[0] + 1LL);
  }
  else
  {
    result = sub_16E7BC0((__int64 *)&v34);
    v18 = v32[0];
    if ( (_QWORD *)v32[0] != v33 )
      return (void *)j_j___libc_free_0(v18, v33[0] + 1LL);
  }
  return result;
}
