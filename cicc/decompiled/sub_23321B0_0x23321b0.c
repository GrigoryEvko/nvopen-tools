// Function: sub_23321B0
// Address: 0x23321b0
//
__int64 __fastcall sub_23321B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // r8
  int v13; // r10d
  unsigned int i; // eax
  __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  _BYTE v22[8]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-88h]
  char v24; // [rsp+2Ch] [rbp-74h]
  unsigned __int64 v25; // [rsp+48h] [rbp-58h]
  char v26; // [rsp+5Ch] [rbp-44h]

  sub_232FC90(a1);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v11 = a6;
  v12 = a5;
  if ( (_DWORD)v9 )
  {
    v13 = 1;
    for ( i = (v9 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)qword_50059C8 >> 9) ^ ((unsigned int)qword_50059C8 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v16 )
    {
      v15 = v10 + 24LL * i;
      if ( *(__int64 **)v15 == qword_50059C8 && a3 == *(_QWORD *)(v15 + 8) )
        break;
      if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
        goto LABEL_13;
      v16 = v13 + i;
      ++v13;
    }
    if ( v15 != v10 + 24 * v9 && *(_QWORD *)(*(_QWORD *)(v15 + 16) + 24LL) )
    {
      sub_283EA00(v22, a2 + 8);
      sub_BBADB0(a1, (__int64)v22, v17, v18);
      if ( !v26 )
        _libc_free(v25);
      if ( !v24 )
        _libc_free(v23);
    }
  }
LABEL_13:
  sub_2330EB0(a1, (__int64)qword_50059C8, v9, v10, v12, v11);
  return a1;
}
