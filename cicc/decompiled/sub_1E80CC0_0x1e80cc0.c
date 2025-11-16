// Function: sub_1E80CC0
// Address: 0x1e80cc0
//
__int64 __fastcall sub_1E80CC0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rcx
  _BYTE *v7; // rdi
  unsigned int v8; // r12d
  __int64 v9; // rsi
  unsigned int v10; // r12d
  __int64 v11; // r8
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r9
  __int16 v15; // ax
  int v16; // eax
  int v18; // eax
  int v19; // r10d
  _QWORD v20[2]; // [rsp+0h] [rbp-40h] BYREF
  _BYTE v21[48]; // [rsp+10h] [rbp-30h] BYREF

  v3 = *a1;
  v4 = *(_QWORD *)(*a1 + 440LL);
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 232) + 96LL)
                 + 0xFFFFFFFDD1745D18LL * (unsigned int)((__int64)(a1[1] - *(_QWORD *)(*a1 + 8LL)) >> 3));
  v20[0] = v21;
  v20[1] = 0x100000000LL;
  if ( v5 )
  {
    sub_1E7F930(*(_QWORD *)(a2 + 32), *(_DWORD *)(a2 + 40), (__int64)v20, v5, *(_QWORD *)(v4 + 256), (int)v20);
    v7 = (_BYTE *)v20[0];
    v3 = *a1;
  }
  else
  {
    v7 = v21;
  }
  v8 = *(_DWORD *)(v3 + 400);
  v9 = *(_QWORD *)v7;
  if ( v8 )
  {
    v10 = v8 - 1;
    v11 = *(_QWORD *)(v3 + 384);
    v12 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v13 = (__int64 *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( v9 == *v13 )
    {
LABEL_5:
      v8 = *((_DWORD *)v13 + 2);
    }
    else
    {
      v18 = 1;
      while ( v14 != -8 )
      {
        v19 = v18 + 1;
        v12 = v10 & (v18 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
          goto LABEL_5;
        v18 = v19;
      }
      v8 = 0;
    }
  }
  v15 = **(_WORD **)(v9 + 16);
  switch ( v15 )
  {
    case 0:
    case 8:
    case 10:
    case 14:
    case 15:
    case 45:
      break;
    default:
      switch ( v15 )
      {
        case 2:
        case 3:
        case 4:
        case 6:
        case 9:
        case 12:
        case 13:
        case 17:
        case 18:
          goto LABEL_9;
        default:
          v16 = sub_1F4BB70(*(_QWORD *)(v3 + 440) + 272LL, v9, *((unsigned int *)v7 + 2), a2, *((unsigned int *)v7 + 3));
          v7 = (_BYTE *)v20[0];
          v8 += v16;
          break;
      }
      break;
  }
LABEL_9:
  if ( v7 != v21 )
    _libc_free((unsigned __int64)v7);
  return v8;
}
