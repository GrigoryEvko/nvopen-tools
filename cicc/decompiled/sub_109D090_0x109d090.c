// Function: sub_109D090
// Address: 0x109d090
//
__int64 __fastcall sub_109D090(__int64 *a1, __int64 a2, unsigned int a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64, __int64, _QWORD); // rax
  int v10; // ebx
  __int64 v11; // r12
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  _BYTE v19[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v8 = a1[10];
  v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v8 + 48LL);
  if ( a4 )
  {
    v10 = a3;
    v11 = v9(v8, 12, a2, a3);
    if ( v11 )
      return v11;
  }
  else
  {
    v11 = v9(v8, 12, a2, *((unsigned int *)a1 + 26));
    if ( v11 )
      return v11;
    v10 = *((_DWORD *)a1 + 26);
  }
  v20 = 257;
  v13 = sub_B50340(12, a2, (__int64)v19, 0, 0);
  v11 = v13;
  if ( a6 || (a6 = a1[12]) != 0 )
    sub_B99FD0(v13, 3u, a6);
  sub_B45150(v11, v10);
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a5,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  while ( v15 != v14 )
  {
    v16 = *(_QWORD *)(v14 + 8);
    v17 = *(_DWORD *)v14;
    v14 += 16;
    sub_B99FD0(v11, v17, v16);
  }
  return v11;
}
