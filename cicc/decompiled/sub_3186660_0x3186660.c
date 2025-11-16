// Function: sub_3186660
// Address: 0x3186660
//
__int64 *__fastcall sub_3186660(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  int v14; // r9d

  *a1 = 0;
  v5 = *(unsigned int *)(a2 + 112);
  v6 = *(_QWORD *)(a2 + 96);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a3 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
      {
        v10 = v8[1];
        v8[1] = 0;
        v11 = *a1;
        *a1 = v10;
        if ( v11 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
        v12 = v8[1];
        if ( v12 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
        *v8 = -8192;
        --*(_DWORD *)(a2 + 104);
        ++*(_DWORD *)(a2 + 108);
      }
    }
    else
    {
      v14 = 1;
      while ( v9 != -4096 )
      {
        v7 = (v5 - 1) & (v14 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a3 == *v8 )
          goto LABEL_3;
        ++v14;
      }
    }
  }
  return a1;
}
