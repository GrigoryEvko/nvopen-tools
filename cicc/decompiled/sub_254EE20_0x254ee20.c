// Function: sub_254EE20
// Address: 0x254ee20
//
__int64 __fastcall sub_254EE20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r15
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // r14
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  int v17; // r11d
  unsigned int i; // eax
  __int64 v19; // r8
  unsigned int v21; // eax
  __int64 v22; // r14

  v4 = (_QWORD *)(a2 + 72);
  if ( !sub_25096F0((_QWORD *)(a2 + 72)) )
    goto LABEL_12;
  v9 = *(_QWORD *)(a3 + 208);
  v10 = sub_25096F0((_QWORD *)(a2 + 72));
  v11 = *(_QWORD *)(v9 + 240);
  v12 = *(_QWORD *)v11;
  if ( !*(_QWORD *)v11 )
    goto LABEL_12;
  if ( *(_BYTE *)(v11 + 16) )
  {
    v15 = *(unsigned int *)(v12 + 88);
    v16 = *(_QWORD *)(v12 + 72);
    if ( !(_DWORD)v15 )
      goto LABEL_12;
    v17 = 1;
    for ( i = (v15 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4FDBCC8 >> 9) ^ ((unsigned int)&unk_4FDBCC8 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = (v15 - 1) & v21 )
    {
      v19 = v16 + 24LL * i;
      if ( *(_UNKNOWN **)v19 == &unk_4FDBCC8 && v10 == *(_QWORD *)(v19 + 8) )
        break;
      if ( *(_QWORD *)v19 == -4096 && *(_QWORD *)(v19 + 8) == -4096 )
        goto LABEL_12;
      v21 = v17 + i;
      ++v17;
    }
    if ( v19 == v16 + 24 * v15 )
      goto LABEL_12;
    v22 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL);
    if ( !v22 )
      goto LABEL_12;
    v13 = (__int64 *)(v22 + 8);
  }
  else
  {
    v13 = (__int64 *)(sub_BC1CD0(v12, &unk_4FDBCC8, v10) + 8);
  }
  if ( !a4 )
  {
LABEL_12:
    sub_AADB10(a1, *(_DWORD *)(a2 + 96), 1);
    return a1;
  }
  v14 = sub_250D070(v4);
  sub_22CE1E0(a1, v13, v14, a4, 0);
  return a1;
}
