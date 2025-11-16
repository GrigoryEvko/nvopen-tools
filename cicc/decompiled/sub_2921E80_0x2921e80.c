// Function: sub_2921E80
// Address: 0x2921e80
//
__int64 __fastcall sub_2921E80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  char *v10; // r15
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rsi
  int v13; // eax
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v18; // rsi
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  char *v26; // r15
  _QWORD v27[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v28; // [rsp+10h] [rbp-40h]

  v27[0] = 4;
  v8 = *(_QWORD *)(a1 + 16);
  v27[1] = 0;
  v28 = a2;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)v27);
  v9 = *(unsigned int *)(v8 + 224);
  v10 = (char *)v27;
  v11 = *(_QWORD *)(v8 + 216);
  v12 = v9 + 1;
  v13 = *(_DWORD *)(v8 + 224);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 228) )
  {
    v25 = v8 + 216;
    if ( v11 > (unsigned __int64)v27 || (unsigned __int64)v27 >= v11 + 24 * v9 )
    {
      sub_D6B130(v25, v12, v9, v11, a5, a6);
      v9 = *(unsigned int *)(v8 + 224);
      v11 = *(_QWORD *)(v8 + 216);
      v13 = *(_DWORD *)(v8 + 224);
    }
    else
    {
      v26 = (char *)v27 - v11;
      sub_D6B130(v25, v12, v9, v11, a5, a6);
      v11 = *(_QWORD *)(v8 + 216);
      v9 = *(unsigned int *)(v8 + 224);
      v10 = &v26[v11];
      v13 = *(_DWORD *)(v8 + 224);
    }
  }
  v14 = (unsigned __int64 *)(v11 + 24 * v9);
  if ( v14 )
  {
    *v14 = 4;
    v15 = *((_QWORD *)v10 + 2);
    v14[1] = 0;
    v14[2] = v15;
    if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
      sub_BD6050(v14, *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
    v13 = *(_DWORD *)(v8 + 224);
  }
  *(_DWORD *)(v8 + 224) = v13 + 1;
  if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
    sub_BD60C0(v27);
  if ( sub_BD2BE0(a2) )
  {
    sub_BD5FC0(*(_QWORD *)(a1 + 152), a2);
  }
  else if ( !(unsigned __int8)sub_B46A50(a2) )
  {
    v16 = *(_QWORD *)(a1 + 112);
    if ( v16 == *(_QWORD *)(a1 + 40) )
    {
      v18 = *(_QWORD *)(a1 + 120);
      if ( v18 == *(_QWORD *)(a1 + 48) )
      {
        v19 = sub_ACD640(*(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL), v18 - v16, 0);
        v20 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
          v20 = **(_QWORD **)(v20 + 16);
        v21 = sub_BCE3C0(*(__int64 **)(a1 + 248), *(_DWORD *)(v20 + 8) >> 8);
        v22 = sub_291C360((__int64 *)a1, a1 + 176, v21);
        v23 = *(_QWORD *)(a2 - 32);
        if ( !v23 || *(_BYTE *)v23 || *(_QWORD *)(v23 + 24) != *(_QWORD *)(a2 + 80) )
          BUG();
        v24 = a1 + 176;
        if ( *(_DWORD *)(v23 + 36) == 211 )
          sub_B34940(v24, v22, v19);
        else
          sub_B349D0(v24, v22, v19);
      }
    }
  }
  return 1;
}
