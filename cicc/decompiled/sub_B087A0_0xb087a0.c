// Function: sub_B087A0
// Address: 0xb087a0
//
__int64 __fastcall sub_B087A0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r14d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 v11; // r13
  _BYTE *v12; // rdi
  unsigned __int8 v13; // al
  _BYTE *v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r14d
  unsigned int v17; // edx
  __int64 *v18; // rsi
  int v19; // r8d
  __int64 *v20; // r9
  int v21; // eax
  __int64 v22; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v23; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+18h] [rbp-48h] BYREF
  int v25; // [rsp+20h] [rbp-40h] BYREF
  int v26[15]; // [rsp+24h] [rbp-3Ch] BYREF

  v22 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v22;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v23 = 0;
    goto LABEL_7;
  }
  v11 = *(_QWORD *)(a3 + 8);
  v12 = (_BYTE *)(a1 - 16);
  v13 = *(_BYTE *)(a1 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_BYTE **)(a1 - 32);
  else
    v14 = &v12[-8 * ((v13 >> 2) & 0xF)];
  v23 = (__int64 *)*((_QWORD *)v14 + 1);
  v15 = a1;
  if ( *(_BYTE *)a1 != 16 )
    v15 = *(_QWORD *)sub_A17150(v12);
  v24 = v15;
  v16 = v4 - 1;
  v25 = *(_DWORD *)(a1 + 4);
  v26[0] = *(unsigned __int16 *)(a1 + 16);
  v8 = v22;
  v17 = v16 & sub_AF7510((__int64 *)&v23, &v24, &v25, v26);
  v18 = (__int64 *)(v11 + 8LL * v17);
  result = *v18;
  if ( v22 != *v18 )
  {
    v19 = 1;
    v7 = 0;
    while ( result != -4096 )
    {
      if ( result != -8192 || v7 )
        v18 = v7;
      v17 = v16 & (v19 + v17);
      v20 = (__int64 *)(v11 + 8LL * v17);
      result = *v20;
      if ( *v20 == v22 )
        return result;
      ++v19;
      v7 = v18;
      v18 = (__int64 *)(v11 + 8LL * v17);
    }
    v21 = *(_DWORD *)(a3 + 16);
    v4 = *(_DWORD *)(a3 + 24);
    if ( !v7 )
      v7 = v18;
    ++*(_QWORD *)a3;
    v9 = v21 + 1;
    v23 = v7;
    if ( 4 * v9 < 3 * v4 )
    {
      if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
        goto LABEL_9;
      v6 = v4;
LABEL_8:
      sub_B084F0(a3, v6);
      sub_AFE170(a3, &v22, &v23);
      v7 = v23;
      v8 = v22;
      v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
      *(_DWORD *)(a3 + 16) = v9;
      if ( *v7 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v7 = v8;
      return v22;
    }
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  return result;
}
