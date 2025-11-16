// Function: sub_21861D0
// Address: 0x21861d0
//
__int64 __fastcall sub_21861D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5)
{
  __int64 v7; // r13
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rax
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 v11; // r13
  __int64 **v12; // rax
  __int64 *v13; // rbx
  unsigned int i; // r13d
  __int64 v15; // r13
  unsigned int j; // ebx
  int v18; // edx
  unsigned int v19; // ecx
  unsigned int v20; // eax
  void *v21; // rdi
  int v22; // edx
  unsigned int v23; // eax
  unsigned int v24; // ecx
  unsigned int v25; // edx
  int v26; // r13d
  unsigned int v27; // eax
  unsigned int v28; // edx
  int v29; // r14d
  unsigned int v30; // eax
  int v31; // [rsp+1Ch] [rbp-64h] BYREF
  _BYTE v32[96]; // [rsp+20h] [rbp-60h] BYREF

  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a3 + 40);
  v7 = *(_QWORD *)(a3 + 16);
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 112LL);
  if ( v8 == sub_214AB90 )
    v9 = v7 + 320;
  else
    v9 = v8(*(_QWORD *)(a3 + 16));
  *(_QWORD *)(a1 + 72) = v9;
  v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL);
  if ( v10 == sub_2163CB0 )
    v11 = v7 + 264;
  else
    v11 = v10(v7);
  *(_QWORD *)(a1 + 88) = v11;
  if ( !a5 )
  {
    v18 = *(_DWORD *)(a1 + 48);
    ++*(_QWORD *)(a1 + 32);
    if ( v18 )
    {
      v19 = 4 * v18;
      v20 = *(_DWORD *)(a1 + 56);
      if ( (unsigned int)(4 * v18) < 0x40 )
        v19 = 64;
      if ( v19 >= v20 )
        goto LABEL_17;
      v28 = v18 - 1;
      if ( v28 )
      {
        _BitScanReverse(&v28, v28);
        v29 = 1 << (33 - (v28 ^ 0x1F));
        if ( v29 < 64 )
          v29 = 64;
        if ( v20 == v29 )
          goto LABEL_50;
      }
      else
      {
        v29 = 64;
      }
      j___libc_free_0(*(_QWORD *)(a1 + 40));
      v30 = sub_217D900(v29);
      *(_DWORD *)(a1 + 56) = v30;
      if ( v30 )
      {
        *(_QWORD *)(a1 + 40) = sub_22077B0(4LL * v30);
LABEL_50:
        sub_217F260(a1 + 32);
LABEL_20:
        v22 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        if ( !v22 )
        {
          if ( !*(_DWORD *)(a1 + 20) )
            goto LABEL_10;
          v23 = *(_DWORD *)(a1 + 24);
          if ( v23 <= 0x40 )
            goto LABEL_23;
          j___libc_free_0(*(_QWORD *)(a1 + 8));
          *(_DWORD *)(a1 + 24) = 0;
LABEL_43:
          *(_QWORD *)(a1 + 8) = 0;
LABEL_25:
          *(_QWORD *)(a1 + 16) = 0;
          goto LABEL_10;
        }
        v24 = 4 * v22;
        v23 = *(_DWORD *)(a1 + 24);
        if ( (unsigned int)(4 * v22) < 0x40 )
          v24 = 64;
        if ( v23 <= v24 )
        {
LABEL_23:
          if ( 4LL * v23 )
            memset(*(void **)(a1 + 8), 255, 4LL * v23);
          goto LABEL_25;
        }
        v25 = v22 - 1;
        if ( v25 )
        {
          _BitScanReverse(&v25, v25);
          v26 = 1 << (33 - (v25 ^ 0x1F));
          if ( v26 < 64 )
            v26 = 64;
          if ( v23 == v26 )
            goto LABEL_41;
        }
        else
        {
          v26 = 64;
        }
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        v27 = sub_217D900(v26);
        *(_DWORD *)(a1 + 24) = v27;
        if ( !v27 )
          goto LABEL_43;
        *(_QWORD *)(a1 + 8) = sub_22077B0(4LL * v27);
LABEL_41:
        sub_217F260(a1);
        goto LABEL_10;
      }
    }
    else
    {
      if ( !*(_DWORD *)(a1 + 52) )
        goto LABEL_20;
      v20 = *(_DWORD *)(a1 + 56);
      if ( v20 <= 0x40 )
      {
LABEL_17:
        v21 = *(void **)(a1 + 40);
        if ( 4LL * v20 )
          memset(v21, 255, 4LL * v20);
        goto LABEL_19;
      }
      j___libc_free_0(*(_QWORD *)(a1 + 40));
      *(_DWORD *)(a1 + 56) = 0;
    }
    *(_QWORD *)(a1 + 40) = 0;
LABEL_19:
    *(_QWORD *)(a1 + 48) = 0;
    goto LABEL_20;
  }
  v12 = (__int64 **)sub_1C01EA0((__int64)a5, *(_QWORD *)(a1 + 64));
  v13 = *v12;
  for ( i = sub_217D950(*v12, 0, *((_DWORD *)*v12 + 4)); i != -1; i = sub_217D950(v13, i + 1, *((_DWORD *)v13 + 4)) )
  {
    v31 = sub_2185E80((__int64)a5, i);
    sub_217F7B0((__int64)v32, a1 + 32, &v31);
  }
  v15 = *(_QWORD *)(sub_1C01EA0((__int64)a5, *(_QWORD *)(a1 + 64)) + 8);
  for ( j = sub_217D950((__int64 *)v15, 0, *(_DWORD *)(v15 + 16));
        j != -1;
        j = sub_217D950((__int64 *)v15, j + 1, *(_DWORD *)(v15 + 16)) )
  {
    v31 = sub_2185E80((__int64)a5, j);
    sub_217F7B0((__int64)v32, a1, &v31);
  }
LABEL_10:
  if ( byte_4FD2DA0 )
    return sub_2184890(a1, a5);
  else
    return sub_2180B10(a1);
}
