// Function: sub_2332050
// Address: 0x2332050
//
__int64 __fastcall sub_2332050(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int i; // eax
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rcx
  _BYTE v17[8]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-88h]
  char v19; // [rsp+2Ch] [rbp-74h]
  unsigned __int64 v20; // [rsp+48h] [rbp-58h]
  char v21; // [rsp+5Ch] [rbp-44h]

  sub_232FC90(a1);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  if ( (_DWORD)v9 )
  {
    v8 = 1;
    for ( i = (v9 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_500CD08 >> 9) ^ ((unsigned int)&unk_500CD08 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v13 )
    {
      v12 = v10 + 24LL * i;
      v7 = *(void **)v12;
      if ( *(_UNKNOWN **)v12 == &unk_500CD08 && a3 == *(_QWORD *)(v12 + 8) )
        break;
      if ( v7 == (void *)-4096LL && *(_QWORD *)(v12 + 8) == -4096 )
        goto LABEL_13;
      v13 = v8 + i;
      v8 = (unsigned int)(v8 + 1);
    }
    if ( v12 != v10 + 24 * v9 && *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) )
    {
      sub_BC2570((__int64)v17, (_QWORD *)(a2 + 8), a3, a4);
      sub_BBADB0(a1, (__int64)v17, v14, v15);
      if ( !v21 )
        _libc_free(v20);
      if ( !v19 )
        _libc_free(v18);
    }
  }
LABEL_13:
  sub_2330EB0(a1, (__int64)&unk_500CD08, v9, v10, (__int64)v7, v8);
  return a1;
}
