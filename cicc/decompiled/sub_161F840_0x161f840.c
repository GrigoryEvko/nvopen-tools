// Function: sub_161F840
// Address: 0x161f840
//
void __fastcall sub_161F840(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 *v11; // rdi
  __int64 v12; // rcx
  int v13; // edi
  int v14; // r9d

  *(_DWORD *)(a2 + 8) = 0;
  v4 = *(_QWORD *)(a1 + 48);
  if ( !v4 )
    goto LABEL_5;
  v5 = 0;
  if ( !*(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, a2 + 16, 0, 16);
    v5 = 16LL * *(unsigned int *)(a2 + 8);
  }
  v6 = *(_QWORD *)a2;
  *(_QWORD *)(v6 + v5) = 0;
  *(_QWORD *)(v6 + v5 + 8) = v4;
  ++*(_DWORD *)(a2 + 8);
  if ( *(__int16 *)(a1 + 18) < 0 )
  {
LABEL_5:
    v7 = *(_QWORD *)sub_16498A0(a1);
    v8 = *(_QWORD *)(v7 + 2712);
    v9 = *(unsigned int *)(v7 + 2728);
    if ( (_DWORD)v9 )
    {
      v10 = (v9 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v11 = (__int64 *)(v8 + 56LL * v10);
      v12 = *v11;
      if ( a1 == *v11 )
      {
LABEL_7:
        sub_161F690(v11 + 1, a2);
        return;
      }
      v13 = 1;
      while ( v12 != -8 )
      {
        v14 = v13 + 1;
        v10 = (v9 - 1) & (v13 + v10);
        v11 = (__int64 *)(v8 + 56LL * v10);
        v12 = *v11;
        if ( a1 == *v11 )
          goto LABEL_7;
        v13 = v14;
      }
    }
    v11 = (__int64 *)(v8 + 56 * v9);
    goto LABEL_7;
  }
}
