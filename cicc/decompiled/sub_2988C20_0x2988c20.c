// Function: sub_2988C20
// Address: 0x2988c20
//
void __fastcall sub_2988C20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v14; // rsi
  int v15; // r9d
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(unsigned int *)(a1 + 904);
  v5 = *(_QWORD *)(a1 + 888);
  if ( !(_DWORD)v4 )
    return;
  v7 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a3 == *v8 )
  {
LABEL_3:
    if ( v8 == (__int64 *)(v5 + 16 * v4) )
      return;
    v10 = v8[1];
    v16[0] = v10;
    if ( v10 )
    {
      v11 = a2 + 48;
      sub_B96E90((__int64)v16, v10, 1);
      v12 = *(_QWORD *)(a2 + 48);
      if ( !v12 )
        goto LABEL_7;
    }
    else
    {
      v12 = *(_QWORD *)(a2 + 48);
      v11 = a2 + 48;
      if ( !v12 )
      {
LABEL_9:
        v14 = v8[1];
        if ( v14 )
          sub_B91220((__int64)(v8 + 1), v14);
        *v8 = -8192;
        --*(_DWORD *)(a1 + 896);
        ++*(_DWORD *)(a1 + 900);
        return;
      }
    }
    sub_B91220(v11, v12);
LABEL_7:
    v13 = (unsigned __int8 *)v16[0];
    *(_QWORD *)(a2 + 48) = v16[0];
    if ( v13 )
      sub_B976B0((__int64)v16, v13, v11);
    goto LABEL_9;
  }
  v15 = 1;
  while ( v9 != -4096 )
  {
    v7 = (v4 - 1) & (v15 + v7);
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a3 == *v8 )
      goto LABEL_3;
    ++v15;
  }
}
