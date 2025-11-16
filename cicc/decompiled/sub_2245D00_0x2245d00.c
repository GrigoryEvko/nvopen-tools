// Function: sub_2245D00
// Address: 0x2245d00
//
__int64 __fastcall sub_2245D00(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, wchar_t a6, long double a7)
{
  __int64 v7; // rsi
  int v8; // eax
  int v9; // r12d
  void *v10; // rsp
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r12
  const wchar_t *v14; // rdi
  int v16; // esi
  __int64 v17; // [rsp-1Eh] [rbp-F0h]
  __int64 v18; // [rsp-16h] [rbp-E8h]
  char v19[15]; // [rsp-Eh] [rbp-E0h] BYREF
  char *v20; // [rsp+4Ah] [rbp-88h]
  __int64 v21; // [rsp+52h] [rbp-80h]
  wchar_t v22; // [rsp+5Ah] [rbp-78h]
  int v23; // [rsp+5Eh] [rbp-74h]
  __int64 v24; // [rsp+62h] [rbp-70h]
  __int64 v25; // [rsp+6Ah] [rbp-68h]
  __int64 v26; // [rsp+72h] [rbp-60h]
  __int64 v27; // [rsp+7Ah] [rbp-58h]
  char v28; // [rsp+91h] [rbp-41h] BYREF
  volatile signed __int32 *v29; // [rsp+92h] [rbp-40h] BYREF
  const wchar_t *v30[7]; // [rsp+9Ah] [rbp-38h] BYREF

  v26 = a1;
  v25 = a2;
  v7 = a5 + 208;
  v24 = a3;
  v23 = a4;
  v21 = a5;
  v22 = a6;
  sub_2208E20(&v29, (volatile signed __int32 **)(a5 + 208));
  v27 = sub_2243120(&v29, v7);
  v30[0] = (const wchar_t *)sub_2208E60(&v29, v7);
  v8 = sub_2218500((__int64)v30, v19, 64, "%.*Lf", 0, a7);
  if ( v8 > 63 )
  {
    v9 = v8 + 1;
    v10 = alloca(v8 + 1 + 8LL);
    v30[0] = (const wchar_t *)sub_2208E60(v18, v17);
    v8 = sub_2218500((__int64)v30, v19, v9, "%.*Lf", 0, a7);
  }
  v11 = v8;
  v20 = &v28;
  sub_2216180(v30, v8, 0);
  if ( *(v30[0] - 2) >= 0 )
    sub_22163D0(v30);
  (*(void (__fastcall **)(__int64, char *, char *))(*(_QWORD *)v27 + 88LL))(v27, v19, &v19[v11]);
  if ( (_BYTE)v23 )
    v12 = sub_2244FC0(v26, v25, v24, v21, v22, v30);
  else
    v12 = sub_2245660(v26, v25, v24, v21, v22, v30);
  v13 = v12;
  v14 = v30[0] - 6;
  if ( v30[0] - 6 != (const wchar_t *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v16 = _InterlockedExchangeAdd((volatile signed __int32 *)v30[0] - 2, 0xFFFFFFFF);
    }
    else
    {
      v16 = *(v30[0] - 2);
      *((_DWORD *)v30[0] - 2) = v16 - 1;
    }
    if ( v16 <= 0 )
      j_j___libc_free_0_2((unsigned __int64)v14);
  }
  sub_2209150(&v29);
  return v13;
}
