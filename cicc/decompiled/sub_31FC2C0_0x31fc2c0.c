// Function: sub_31FC2C0
// Address: 0x31fc2c0
//
void __fastcall sub_31FC2C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdx
  int v12; // eax
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // r8
  __int64 v17; // rdi
  void (*v18)(); // rax
  __int64 v19; // rsi
  __int64 v20; // rdi
  void (*v21)(); // rax
  unsigned __int64 v22; // [rsp+8h] [rbp-C8h]
  __int64 v23; // [rsp+18h] [rbp-B8h]
  __int64 v24; // [rsp+20h] [rbp-B0h] BYREF
  char v25; // [rsp+28h] [rbp-A8h]
  char *v26; // [rsp+30h] [rbp-A0h] BYREF
  char v27; // [rsp+38h] [rbp-98h]
  char v28; // [rsp+50h] [rbp-80h]
  char v29; // [rsp+51h] [rbp-7Fh]
  void *base; // [rsp+60h] [rbp-70h] BYREF
  __int64 v31; // [rsp+68h] [rbp-68h]
  _BYTE v32[96]; // [rsp+70h] [rbp-60h] BYREF

  if ( a2[8] )
  {
    v7 = a2[6];
    v27 = 0;
    base = v32;
    v31 = 0xC00000000LL;
    v26 = (char *)(a2 + 4);
    v25 = 0;
  }
  else
  {
    v7 = *a2;
    v8 = *((unsigned int *)a2 + 2);
    v27 = 1;
    base = v32;
    v31 = 0xC00000000LL;
    v26 = (char *)(v7 + 4 * v8);
    v25 = 1;
  }
  v24 = v7;
  sub_31FC180((__int64)&base, (__int64)&v24, (__int64)&v26, 0xC00000000LL, a5, a6);
  v11 = (unsigned int)v31;
  v12 = v31;
  if ( (unsigned int)v31 > 1uLL )
  {
    qsort(base, (4LL * (unsigned int)v31) >> 2, 4u, (__compar_fn_t)sub_31F3C90);
    v12 = v31;
  }
  if ( v12 )
  {
    v13 = 0;
    do
    {
      v14 = sub_31F8790(a1, 4456, v11, v9, v10);
      v15 = (unsigned int)v31;
      v23 = v14;
      v16 = (unsigned int)v31 - v13;
      if ( v16 > 0x3FBD )
      {
        v15 = v13 + 16318;
        v16 = 16318;
      }
      v17 = *(_QWORD *)(a1 + 528);
      v18 = *(void (**)())(*(_QWORD *)v17 + 120LL);
      v29 = 1;
      v26 = "Count";
      v28 = 3;
      if ( v18 != nullsub_98 )
      {
        v22 = v16;
        ((void (__fastcall *)(__int64, char **, __int64))v18)(v17, &v26, 1);
        v17 = *(_QWORD *)(a1 + 528);
        v16 = v22;
      }
      (*(void (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v17 + 536LL))(v17, v16, 4);
      if ( v15 > v13 )
      {
        do
        {
          v20 = *(_QWORD *)(a1 + 528);
          v21 = *(void (**)())(*(_QWORD *)v20 + 120LL);
          v29 = 1;
          v26 = "Inlinee";
          v28 = 3;
          if ( v21 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, char **, __int64))v21)(v20, &v26, 1);
            v20 = *(_QWORD *)(a1 + 528);
          }
          v19 = *((unsigned int *)base + v13++);
          (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v20 + 536LL))(v20, v19, 4);
        }
        while ( v13 != v15 );
      }
      sub_31F8930(a1, v23);
    }
    while ( v13 < (unsigned int)v31 );
  }
  if ( base != v32 )
    _libc_free((unsigned __int64)base);
}
