// Function: sub_23A1380
// Address: 0x23a1380
//
__int64 __fastcall sub_23A1380(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // r13
  void (__fastcall **v8)(__int64, __int64, __int64 *); // rax
  __int64 v9; // rdi
  void (*v10)(); // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  void (*v17)(); // rax
  __int64 v19; // rax

  v6 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v8 = (void (__fastcall **)(__int64, __int64, __int64 *))(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  v9 = *a2;
  if ( *a2 )
  {
    v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
    if ( v10 != nullsub_1476 )
    {
      ((void (__fastcall *)(__int64, __int64))v10)(v9, a1);
      v19 = *(unsigned int *)(a1 + 8);
      if ( v19 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v8 = (void (__fastcall **)(__int64, __int64, __int64 *))(*(_QWORD *)a1 + 8 * v19);
      }
      else
      {
        sub_C8D5F0(a1, v6, v19 + 1, 8u, a5, a6);
        v8 = (void (__fastcall **)(__int64, __int64, __int64 *))(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      }
    }
  }
  *v8 = sub_2361CE0;
  v11 = *(unsigned int *)(a1 + 12);
  v12 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v12;
  if ( v12 + 1 > v11 )
  {
    sub_C8D5F0(a1, v6, v12 + 1, 8u, a5, a6);
    v12 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v12) = sub_2362120;
  v13 = *(unsigned int *)(a1 + 12);
  v14 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v14;
  if ( v14 + 1 > v13 )
  {
    sub_C8D5F0(a1, v6, v14 + 1, 8u, a5, a6);
    v14 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v14) = sub_2362200;
  v15 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v15;
  if ( byte_4FDDAE8 )
  {
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, v6, v15 + 1, 8u, a5, a6);
      v15 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v15) = sub_2396EC0;
    ++*(_DWORD *)(a1 + 8);
  }
  v16 = *a2;
  if ( !*a2 )
    return a1;
  v17 = *(void (**)())(*(_QWORD *)v16 + 128LL);
  if ( v17 == nullsub_1477 )
    return a1;
  ((void (__fastcall *)(__int64, __int64))v17)(v16, a1);
  return a1;
}
