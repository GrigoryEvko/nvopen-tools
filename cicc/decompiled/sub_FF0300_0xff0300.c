// Function: sub_FF0300
// Address: 0xff0300
//
__int64 __fastcall sub_FF0300(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rbp
  __int64 v5; // rdx
  __int64 v6; // r8
  int v7; // r11d
  unsigned int i; // eax
  __int64 v9; // rdi
  unsigned int v10; // eax
  _QWORD *v12; // rsi
  unsigned __int64 v13; // rax
  unsigned int v14; // edx
  unsigned int v15; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v16; // [rsp-8h] [rbp-8h]

  v5 = *(unsigned int *)(a1 + 56);
  v6 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v5 )
  {
    v7 = 1;
    for ( i = (v5 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (756364221 * a3)); ; i = (v5 - 1) & v10 )
    {
      v9 = v6 + 24LL * i;
      if ( a2 == *(_QWORD *)v9 && a3 == *(_DWORD *)(v9 + 8) )
        break;
      if ( *(_QWORD *)v9 == -4096 && *(_DWORD *)(v9 + 8) == -1 )
        goto LABEL_10;
      v10 = v7 + i;
      ++v7;
    }
    if ( v9 != v6 + 24 * v5 )
      return *(unsigned int *)(v9 + 16);
  }
LABEL_10:
  v16 = v3;
  v12 = (_QWORD *)(a2 + 48);
  v13 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v13 == v12 )
    goto LABEL_15;
  if ( !v13 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_15:
    v14 = 0;
  else
    v14 = sub_B46E30(v13 - 24);
  sub_F02DB0(&v15, 1u, v14);
  return v15;
}
