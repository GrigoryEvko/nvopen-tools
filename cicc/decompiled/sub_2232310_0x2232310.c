// Function: sub_2232310
// Address: 0x2232310
//
__int64 __fastcall sub_2232310(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, int a6, long double a7)
{
  __int64 v7; // rsi
  int v8; // eax
  int v9; // r12d
  void *v10; // rsp
  size_t v11; // r12
  void *v12; // r14
  char v13; // al
  _BYTE *(__fastcall *v14)(__int64, char *, char *, void *); // rax
  __int64 v15; // rax
  __int64 v16; // r12
  char *v17; // rdi
  int v19; // edx
  __int64 v20; // [rsp-1Eh] [rbp-F0h]
  __int64 v21; // [rsp-16h] [rbp-E8h]
  char v22[15]; // [rsp-Eh] [rbp-E0h] BYREF
  char *v23; // [rsp+42h] [rbp-90h]
  char *v24; // [rsp+4Ah] [rbp-88h]
  __int64 v25; // [rsp+52h] [rbp-80h]
  int v26; // [rsp+5Ah] [rbp-78h]
  int v27; // [rsp+5Eh] [rbp-74h]
  __int64 v28; // [rsp+62h] [rbp-70h]
  __int64 v29; // [rsp+6Ah] [rbp-68h]
  __int64 v30; // [rsp+72h] [rbp-60h]
  _BYTE *v31; // [rsp+7Ah] [rbp-58h]
  char v32; // [rsp+91h] [rbp-41h] BYREF
  volatile signed __int32 *v33; // [rsp+92h] [rbp-40h] BYREF
  void *dest[7]; // [rsp+9Ah] [rbp-38h] BYREF

  v30 = a1;
  v29 = a2;
  v7 = a5 + 208;
  v28 = a3;
  v27 = a4;
  v25 = a5;
  v26 = a6;
  sub_2208E20(&v33, (volatile signed __int32 **)(a5 + 208));
  v31 = (_BYTE *)sub_222F790(&v33, v7);
  dest[0] = (void *)sub_2208E60(&v33, v7);
  v8 = sub_2218500((__int64)dest, v22, 64, "%.*Lf", 0, a7);
  if ( v8 > 63 )
  {
    v9 = v8 + 1;
    v10 = alloca(v8 + 1 + 8LL);
    dest[0] = (void *)sub_2208E60(v21, v20);
    v8 = sub_2218500((__int64)dest, v22, v9, "%.*Lf", 0, a7);
  }
  v11 = v8;
  v23 = &v32;
  sub_2215510(dest, v8, 0);
  v12 = dest[0];
  if ( *((int *)dest[0] - 2) >= 0 )
  {
    sub_2215730((volatile signed __int32 **)dest);
    v12 = dest[0];
  }
  v24 = &v22[v11];
  v13 = v31[56];
  if ( v13 == 1 )
    goto LABEL_9;
  if ( !v13 )
    sub_2216D60((__int64)v31);
  v14 = *(_BYTE *(__fastcall **)(__int64, char *, char *, void *))(*(_QWORD *)v31 + 56LL);
  if ( v14 == sub_2216D40 )
  {
LABEL_9:
    if ( v24 != v22 )
      memcpy(v12, v22, v11);
  }
  else
  {
    v14((__int64)v31, v22, v24, v12);
  }
  if ( (_BYTE)v27 )
    v15 = sub_22316F0(v30, v29, v28, v25, v26, (volatile signed __int32 **)dest);
  else
    v15 = sub_2231D00(v30, v29, v28, v25, v26, (volatile signed __int32 **)dest);
  v16 = v15;
  v17 = (char *)dest[0] - 24;
  if ( (char *)dest[0] - 24 != (char *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v19 = _InterlockedExchangeAdd((volatile signed __int32 *)dest[0] - 2, 0xFFFFFFFF);
    }
    else
    {
      v19 = *((_DWORD *)dest[0] - 2);
      *((_DWORD *)dest[0] - 2) = v19 - 1;
    }
    if ( v19 <= 0 )
      j_j___libc_free_0_1((unsigned __int64)v17);
  }
  sub_2209150(&v33);
  return v16;
}
