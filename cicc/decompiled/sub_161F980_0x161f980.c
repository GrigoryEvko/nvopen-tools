// Function: sub_161F980
// Address: 0x161f980
//
void __fastcall sub_161F980(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // edx
  __int64 *v8; // rdi
  __int64 v9; // rcx
  int v10; // edi
  int v11; // r9d

  *(_DWORD *)(a2 + 8) = 0;
  v4 = *(_QWORD *)sub_16498A0(a1);
  v5 = *(_QWORD *)(v4 + 2712);
  v6 = *(unsigned int *)(v4 + 2728);
  if ( (_DWORD)v6 )
  {
    v7 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (__int64 *)(v5 + 56LL * v7);
    v9 = *v8;
    if ( a1 == *v8 )
    {
LABEL_3:
      sub_161F690(v8 + 1, a2);
      return;
    }
    v10 = 1;
    while ( v9 != -8 )
    {
      v11 = v10 + 1;
      v7 = (v6 - 1) & (v10 + v7);
      v8 = (__int64 *)(v5 + 56LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        goto LABEL_3;
      v10 = v11;
    }
  }
  sub_161F690((__int64 *)(v5 + 56 * v6 + 8), a2);
}
