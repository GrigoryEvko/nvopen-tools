// Function: sub_F80960
// Address: 0xf80960
//
__int64 __fastcall sub_F80960(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r9
  __int64 v7; // r13
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  int v10; // ecx
  _QWORD *v11; // rdx
  __int64 v12; // rsi
  __int64 result; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  --*(_DWORD *)(*(_QWORD *)(a1 + 56) + 792LL);
  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)a1;
  if ( v2 )
  {
    sub_A88F30(v3, v2, *(_QWORD *)(a1 + 32), *(_WORD *)(a1 + 40));
  }
  else
  {
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 56) = 0;
    *(_WORD *)(v3 + 64) = 0;
  }
  v4 = *(_QWORD *)(a1 + 48);
  v5 = *(_QWORD *)a1;
  v16[0] = v4;
  if ( v4 && (sub_B96E90((__int64)v16, v4, 1), (v7 = v16[0]) != 0) )
  {
    v8 = *(unsigned int *)(v5 + 8);
    v9 = *(_QWORD **)v5;
    v10 = *(_DWORD *)(v5 + 8);
    v11 = (_QWORD *)(*(_QWORD *)v5 + 16 * v8);
    if ( *(_QWORD **)v5 != v11 )
    {
      while ( *(_DWORD *)v9 )
      {
        v9 += 2;
        if ( v11 == v9 )
          goto LABEL_21;
      }
      v9[1] = v16[0];
      goto LABEL_10;
    }
LABEL_21:
    v14 = *(unsigned int *)(v5 + 12);
    if ( v8 >= v14 )
    {
      v15 = v8 + 1;
      if ( v14 < v15 )
      {
        sub_C8D5F0(v5, (const void *)(v5 + 16), v15, 0x10u, v5 + 16, v6);
        v11 = (_QWORD *)(*(_QWORD *)v5 + 16LL * *(unsigned int *)(v5 + 8));
      }
      *v11 = 0;
      v11[1] = v7;
      v7 = v16[0];
      ++*(_DWORD *)(v5 + 8);
    }
    else
    {
      if ( v11 )
      {
        *(_DWORD *)v11 = 0;
        v11[1] = v7;
        v7 = v16[0];
        v10 = *(_DWORD *)(v5 + 8);
      }
      *(_DWORD *)(v5 + 8) = v10 + 1;
    }
  }
  else
  {
    sub_93FB40(v5, 0);
    v7 = v16[0];
  }
  if ( v7 )
LABEL_10:
    sub_B91220((__int64)v16, v7);
  v12 = *(_QWORD *)(a1 + 48);
  if ( v12 )
    sub_B91220(a1 + 48, v12);
  result = *(_QWORD *)(a1 + 24);
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0((_QWORD *)(a1 + 8));
  return result;
}
