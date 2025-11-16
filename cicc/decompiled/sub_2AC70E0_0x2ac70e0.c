// Function: sub_2AC70E0
// Address: 0x2ac70e0
//
__int64 __fastcall sub_2AC70E0(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4, __int64 a5)
{
  char v6; // bl
  __int64 v7; // r12
  const char *v8; // rdx
  unsigned int v9; // r13d
  __int64 *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 *v24; // r9
  __int64 result; // rax
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 *i; // [rsp+18h] [rbp-88h]
  const char *v32; // [rsp+38h] [rbp-68h] BYREF
  const char *v33[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v6 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  v7 = sub_B47F80((_BYTE *)a2);
  if ( v6 != 7 )
  {
    v33[0] = sub_BD5D20(a2);
    v34 = 773;
    v33[1] = v8;
    v33[2] = ".cloned";
    sub_BD6B50((unsigned __int8 *)v7, v33);
  }
  sub_2AAF930(a3, (unsigned __int8 *)v7);
  v32 = *(const char **)(a2 + 48);
  if ( v32 )
  {
    sub_2AAAFA0((__int64 *)&v32);
    if ( v32 )
    {
      v33[0] = v32;
      sub_2AAAFA0((__int64 *)v33);
      sub_2BF1A90(a5, v33);
      sub_9C6650(v33);
    }
  }
  v9 = 0;
  sub_9C6650(&v32);
  v10 = *(__int64 **)(a3 + 48);
  for ( i = &v10[*(unsigned int *)(a3 + 56)]; i != v10; ++v10 )
  {
    v33[0] = *(const char **)a4;
    v15 = *v10;
    if ( (unsigned __int8)sub_2AAA120(*v10) )
    {
      LODWORD(v33[0]) = 0;
      BYTE4(v33[0]) = 0;
    }
    v16 = sub_2BFB120(a5, v15, v33);
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      v11 = *(_QWORD *)(v7 - 8);
    else
      v11 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    v12 = v11 + 32LL * v9;
    if ( *(_QWORD *)v12 )
    {
      v13 = *(_QWORD *)(v12 + 8);
      **(_QWORD **)(v12 + 16) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
    }
    *(_QWORD *)v12 = v16;
    if ( v16 )
    {
      v14 = *(_QWORD *)(v16 + 16);
      *(_QWORD *)(v12 + 8) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = v12 + 8;
      *(_QWORD *)(v12 + 16) = v16 + 16;
      *(_QWORD *)(v16 + 16) = v12;
    }
    ++v9;
  }
  sub_2BF0870(a5, v7, a2);
  v17 = *(__int64 **)(a5 + 904);
  v34 = 257;
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v17[11] + 16LL))(
    v17[11],
    v7,
    v33,
    v17[7],
    v17[8]);
  v18 = *v17;
  v19 = *v17 + 16LL * *((unsigned int *)v17 + 2);
  while ( v19 != v18 )
  {
    v20 = *(_QWORD *)(v18 + 8);
    v21 = *(_DWORD *)v18;
    v18 += 16;
    sub_B99FD0(v7, v21, v20);
  }
  sub_2AC6E90(a5, a3 + 96, v7, a4);
  if ( *(_BYTE *)v7 == 85 )
  {
    v26 = *(_QWORD *)(v7 - 32);
    if ( v26 )
    {
      if ( !*(_BYTE *)v26 )
      {
        v27 = *(_QWORD *)(v7 + 80);
        if ( *(_QWORD *)(v26 + 24) == v27 && (*(_BYTE *)(v26 + 33) & 0x20) != 0 && *(_DWORD *)(v26 + 36) == 11 )
          sub_CFEAE0(*(_QWORD *)(a1 + 56), v7, v22, v27, v23, v24);
      }
    }
  }
  result = *(_QWORD *)(*(_QWORD *)(a3 + 80) + 48LL);
  if ( result )
  {
    if ( *(_BYTE *)(result + 128) )
      return sub_9C95B0(a1 + 312, v7);
  }
  return result;
}
