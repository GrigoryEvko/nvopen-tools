// Function: sub_F15B30
// Address: 0xf15b30
//
__int64 __fastcall sub_F15B30(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v5; // r14
  int v6; // r13d
  bool v7; // zf
  unsigned int v8; // eax
  __int64 v9; // rcx
  int v10; // r9d
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned int i; // eax
  __int64 *v14; // rdx
  __int64 v15; // rsi
  int v16; // eax
  unsigned int v17; // r8d
  char v19; // r8
  unsigned int v20; // eax
  int v21; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v22; // [rsp+10h] [rbp-30h] BYREF
  __int64 v23[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = v16 - 1;
    if ( !v16 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v7 = *((_BYTE *)a2 + 32) == 0;
  v21 = 0;
  if ( !v7 )
    v21 = *((unsigned __int16 *)a2 + 12) | (*((_DWORD *)a2 + 4) << 16);
  v23[0] = a2[5];
  v22 = a2[1];
  v8 = sub_F11290(&v22, &v21, v23);
  v9 = *a2;
  v10 = 1;
  v11 = 0;
  v12 = 0xBF58476D1CE4E5B9LL * (v8 | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32));
  for ( i = v6 & ((v12 >> 31) ^ v12); ; i = v6 & v20 )
  {
    v14 = (__int64 *)(v5 + 56LL * i);
    v15 = *v14;
    if ( *v14 == v9 && a2[1] == v14[1] )
    {
      v19 = *((_BYTE *)a2 + 32);
      if ( v19 == *((_BYTE *)v14 + 32) && (!v19 || a2[2] == v14[2] && a2[3] == v14[3]) && a2[5] == v14[5] )
      {
        *a3 = v14;
        return 1;
      }
    }
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && !v14[1] && *((_BYTE *)v14 + 32) && !v14[2] && !v14[3] && !v14[5] && !v11 )
      v11 = v5 + 56LL * i;
LABEL_26:
    v20 = v10 + i;
    ++v10;
  }
  if ( v14[1] )
    goto LABEL_26;
  v17 = *((unsigned __int8 *)v14 + 32);
  if ( (_BYTE)v17 || v14[5] )
    goto LABEL_26;
  if ( !v11 )
    v11 = v5 + 56LL * i;
  *a3 = v11;
  return v17;
}
