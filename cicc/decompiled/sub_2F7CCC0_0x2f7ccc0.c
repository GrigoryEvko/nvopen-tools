// Function: sub_2F7CCC0
// Address: 0x2f7ccc0
//
__int64 __fastcall sub_2F7CCC0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r12d
  unsigned int v5; // r8d
  bool v7; // zf
  __int64 v9; // r14
  int v10; // r12d
  int v11; // eax
  __int64 v12; // rsi
  int v13; // r9d
  __int64 v14; // rdi
  unsigned int i; // eax
  __int64 *v16; // rcx
  __int64 v17; // rdx
  unsigned int v18; // eax
  char v19; // r8
  int v20; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v21; // [rsp+10h] [rbp-30h] BYREF
  __int64 v22[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *(_BYTE *)(a2 + 24) == 0;
  v20 = 0;
  v9 = *(_QWORD *)(a1 + 8);
  if ( !v7 )
    v20 = *(unsigned __int16 *)(a2 + 16) | (*(_DWORD *)(a2 + 8) << 16);
  v10 = v4 - 1;
  v22[0] = *(_QWORD *)(a2 + 32);
  v21 = *(_QWORD *)a2;
  v11 = sub_F11290(&v21, &v20, v22);
  v12 = *(_QWORD *)a2;
  v13 = 1;
  v14 = 0;
  for ( i = v10 & v11; ; i = v10 & v18 )
  {
    v16 = (__int64 *)(v9 + 56LL * i);
    v17 = *v16;
    if ( v12 == *v16 )
    {
      v19 = *(_BYTE *)(a2 + 24);
      if ( v19 == *((_BYTE *)v16 + 24) )
      {
        if ( v19 && (*(_QWORD *)(a2 + 8) != v16[1] || *(_QWORD *)(a2 + 16) != v16[2]) )
        {
          if ( v17 )
            goto LABEL_9;
          goto LABEL_16;
        }
        if ( *(_QWORD *)(a2 + 32) == v16[4] )
        {
          *a3 = v16;
          return 1;
        }
      }
    }
    if ( v17 )
      goto LABEL_9;
    v5 = *((unsigned __int8 *)v16 + 24);
    if ( !(_BYTE)v5 )
      break;
LABEL_16:
    if ( !v16[1] && !v16[2] && !v16[4] && !v14 )
      v14 = v9 + 56LL * i;
LABEL_9:
    v18 = v13 + i;
    ++v13;
  }
  if ( v16[4] )
    goto LABEL_9;
  if ( !v14 )
    v14 = v9 + 56LL * i;
  *a3 = v14;
  return v5;
}
