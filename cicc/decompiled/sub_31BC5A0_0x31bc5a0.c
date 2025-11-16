// Function: sub_31BC5A0
// Address: 0x31bc5a0
//
__int64 __fastcall sub_31BC5A0(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 v8; // r12
  __int64 v10; // r14
  unsigned __int64 v11; // r12
  __int64 v12; // rdx
  __int64 *v13; // r14
  __int64 *v14; // rbx
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  char *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned __int64 v25; // rdx
  int v26; // eax
  int v27; // [rsp+4h] [rbp-DCh]
  char *v28; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-C8h]
  _DWORD v30[8]; // [rsp+20h] [rbp-C0h] BYREF
  char *v31; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+48h] [rbp-98h]
  _BYTE v33[32]; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v34; // [rsp+70h] [rbp-70h]
  char *v35; // [rsp+78h] [rbp-68h] BYREF
  __int64 v36; // [rsp+80h] [rbp-60h]
  _BYTE v37[88]; // [rsp+88h] [rbp-58h] BYREF

  v7 = *a2;
  v8 = **a2;
  if ( (v8 & 4) != 0 )
  {
    *(_BYTE *)(a1 + 56) = 0;
    return a1;
  }
  v10 = *((unsigned int *)a2 + 2);
  v11 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  v29 = 0x800000000LL;
  v12 = *((unsigned int *)v7 + 2);
  v28 = (char *)v30;
  v13 = &v7[2 * v10];
  v14 = v7 + 2;
  v30[0] = v12;
  LODWORD(v29) = 1;
  if ( v13 == v14 )
  {
    v32 = 0x800000000LL;
    v31 = v33;
  }
  else
  {
    do
    {
      if ( (*v14 & 4) != 0 || v11 != (*v14 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        *(_BYTE *)(a1 + 56) = 0;
        goto LABEL_12;
      }
      v16 = (unsigned int)v29;
      a4 = HIDWORD(v29);
      v17 = *((unsigned int *)v14 + 2);
      v18 = (unsigned int)v29 + 1LL;
      if ( v18 > HIDWORD(v29) )
      {
        v27 = *((_DWORD *)v14 + 2);
        sub_C8D5F0((__int64)&v28, v30, v18, 4u, v17, a6);
        v16 = (unsigned int)v29;
        LODWORD(v17) = v27;
      }
      v12 = (__int64)v28;
      v14 += 2;
      *(_DWORD *)&v28[4 * v16] = v17;
      v15 = v29 + 1;
      LODWORD(v29) = v29 + 1;
    }
    while ( v13 != v14 );
    v31 = v33;
    v32 = 0x800000000LL;
    if ( !v15 )
    {
      v35 = v37;
LABEL_16:
      *(_QWORD *)a1 = v11;
      *(_QWORD *)(a1 + 8) = a1 + 24;
      *(_QWORD *)(a1 + 16) = 0x800000000LL;
      goto LABEL_17;
    }
  }
  sub_31BC430((__int64)&v31, &v28, v12, a4, (__int64)&v31, a6);
  v34 = v11;
  v35 = v37;
  v36 = 0x800000000LL;
  if ( !(_DWORD)v32 )
    goto LABEL_16;
  sub_31BC430((__int64)&v35, &v31, (unsigned int)v32, v20, (__int64)&v31, v21);
  v25 = v34;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  v26 = v36;
  *(_QWORD *)a1 = v25;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  if ( v26 )
    sub_31BC430(a1 + 8, &v35, a1 + 24, v22, v23, v24);
LABEL_17:
  v19 = v35;
  *(_BYTE *)(a1 + 56) = 1;
  if ( v19 != v37 )
    _libc_free((unsigned __int64)v19);
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
LABEL_12:
  if ( v28 != (char *)v30 )
    _libc_free((unsigned __int64)v28);
  return a1;
}
