// Function: sub_904450
// Address: 0x904450
//
void __fastcall sub_904450(__int64 a1, const char *a2)
{
  char *v3; // rax
  const char *v4; // r8
  size_t v5; // r9
  char *v6; // rdx
  _QWORD *v7; // rcx
  unsigned int v8; // r13d
  __int64 v9; // rdx
  __int64 *v10; // rdi
  _QWORD *v11; // r14
  _QWORD *v12; // rbx
  __int64 v13; // rdx
  char *v14; // rax
  char *v15; // rdi
  char *v16; // [rsp-D8h] [rbp-D8h]
  char *v17; // [rsp-C0h] [rbp-C0h] BYREF
  _QWORD *v18; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v19; // [rsp-B0h] [rbp-B0h]
  __int64 v20; // [rsp-A8h] [rbp-A8h]
  char *v21[2]; // [rsp-98h] [rbp-98h] BYREF
  _QWORD v22[2]; // [rsp-88h] [rbp-88h] BYREF
  _QWORD v23[2]; // [rsp-78h] [rbp-78h] BYREF
  _QWORD v24[2]; // [rsp-68h] [rbp-68h] BYREF
  _QWORD v25[2]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v26[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( !a2 )
    return;
  v25[0] = v26;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  strcpy((char *)v26, "\"'");
  v25[1] = 2;
  v23[0] = v24;
  v23[1] = 1;
  LOWORD(v24[0]) = 32;
  v21[0] = (char *)v22;
  v3 = (char *)strlen(a2);
  v4 = a2;
  v17 = v3;
  v5 = (size_t)v3;
  if ( (unsigned __int64)v3 > 0xF )
  {
    v16 = v3;
    v14 = (char *)sub_22409D0(v21, &v17, 0);
    v4 = a2;
    v5 = (size_t)v16;
    v21[0] = v14;
    v15 = v14;
    v22[0] = v17;
LABEL_20:
    memcpy(v15, v4, v5);
    v3 = v17;
    v6 = v21[0];
    goto LABEL_5;
  }
  if ( v3 != (char *)1 )
  {
    if ( !v3 )
    {
      v6 = (char *)v22;
      goto LABEL_5;
    }
    v15 = (char *)v22;
    goto LABEL_20;
  }
  LOBYTE(v22[0]) = *a2;
  v6 = (char *)v22;
LABEL_5:
  v21[1] = v3;
  v3[(_QWORD)v6] = 0;
  sub_9040F0(v21, (__int64)&v18, (__int64)v23, (__int64)v25);
  if ( (_QWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0], v22[0] + 1LL);
  if ( (_QWORD *)v23[0] != v24 )
    j_j___libc_free_0(v23[0], v24[0] + 1LL);
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0], v26[0] + 1LL);
  v7 = v18;
  v8 = 0;
  v9 = 0;
  if ( v19 != v18 )
  {
    do
    {
      v10 = *(__int64 **)(a1 + 8);
      v13 = (__int64)&v7[4 * v9];
      if ( v10 == *(__int64 **)(a1 + 16) )
      {
        sub_8FD760((__m128i **)a1, *(const __m128i **)(a1 + 8), v13);
        v7 = v18;
      }
      else
      {
        if ( v10 )
        {
          *v10 = (__int64)(v10 + 2);
          sub_903680(v10, *(_BYTE **)v13, *(_QWORD *)v13 + *(_QWORD *)(v13 + 8));
          v10 = *(__int64 **)(a1 + 8);
          v7 = v18;
        }
        *(_QWORD *)(a1 + 8) = v10 + 4;
      }
      v11 = v19;
      v12 = v7;
      v9 = ++v8;
    }
    while ( v8 < (unsigned __int64)(((char *)v19 - (char *)v7) >> 5) );
    if ( v7 != v19 )
    {
      do
      {
        if ( (_QWORD *)*v12 != v12 + 2 )
          j_j___libc_free_0(*v12, v12[2] + 1LL);
        v12 += 4;
      }
      while ( v11 != v12 );
    }
  }
  if ( v18 )
    j_j___libc_free_0(v18, v20 - (_QWORD)v18);
}
