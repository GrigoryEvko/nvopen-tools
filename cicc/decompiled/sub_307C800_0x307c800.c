// Function: sub_307C800
// Address: 0x307c800
//
__int64 __fastcall sub_307C800(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 result; // rax
  __int64 v4; // rcx
  unsigned int v5; // r9d
  int v6; // edi
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r11
  __int64 v11; // rbx
  int v12; // eax
  __int64 v13; // rsi
  int v14; // eax
  unsigned int v15; // edx
  int v16; // r8d
  _DWORD *v17; // rax
  int v18; // r9d
  int v19; // eax
  int v20; // ebx
  int v21; // [rsp-1Ch] [rbp-1Ch] BYREF
  __int64 v22; // [rsp-8h] [rbp-8h]

  result = 0;
  if ( **(_QWORD **)a1 == a2 )
    return result;
  v22 = v2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_DWORD *)(v4 + 136);
  v6 = **(_DWORD **)(a1 + 8);
  v7 = *(_QWORD *)(v4 + 120);
  if ( !v5 )
    goto LABEL_9;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v19 = 1;
    while ( v10 != -4096 )
    {
      v20 = v19 + 1;
      v8 = (v5 - 1) & (v19 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_4;
      v19 = v20;
    }
LABEL_9:
    v9 = (__int64 *)(v7 + 16LL * v5);
  }
LABEL_4:
  v11 = v9[1];
  v12 = *(_DWORD *)(v4 + 80);
  v21 = v6;
  v13 = *(_QWORD *)(v4 + 64);
  if ( v12 )
  {
    v14 = v12 - 1;
    v15 = v14 & (37 * v6);
    v16 = *(_DWORD *)(v13 + 8LL * v15);
    if ( v6 == v16 )
    {
LABEL_6:
      v17 = sub_307C5F0(v4 + 56, &v21);
      result = (*(_QWORD *)(*(_QWORD *)(v11 + 24) + 8LL * (*v17 >> 6)) >> *v17) & 1LL;
      if ( (_DWORD)result )
        return result;
    }
    else
    {
      v18 = 1;
      while ( v16 != -1 )
      {
        v15 = v14 & (v18 + v15);
        v16 = *(_DWORD *)(v13 + 8LL * v15);
        if ( v6 == v16 )
          goto LABEL_6;
        ++v18;
      }
    }
  }
  return 0;
}
