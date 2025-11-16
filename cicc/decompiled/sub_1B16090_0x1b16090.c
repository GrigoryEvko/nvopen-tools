// Function: sub_1B16090
// Address: 0x1b16090
//
__int64 __fastcall sub_1B16090(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  _QWORD *v5; // rsi
  int v6; // r12d
  _QWORD *v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v12; // rdx
  __int64 *v13; // [rsp+8h] [rbp-38h]

  v3 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v4 = *(__int64 **)(a1 - 8);
    v13 = &v4[v3];
  }
  else
  {
    v13 = (__int64 *)a1;
    v4 = (__int64 *)(a1 - v3 * 8);
  }
  if ( v4 == v13 )
    return 0;
  v5 = *(_QWORD **)(a2 + 16);
  v6 = 0;
  while ( 1 )
  {
    v9 = *v4;
    if ( *(_BYTE *)(*v4 + 16) <= 0x17u )
      v9 = 0;
    v8 = *(_QWORD **)(a2 + 8);
    if ( v5 == v8 )
    {
      v10 = *(unsigned int *)(a2 + 28);
      v7 = &v5[v10];
      if ( v5 == v7 )
      {
        v12 = (__int64)v5;
      }
      else
      {
        do
        {
          if ( v9 == *v8 )
            break;
          ++v8;
        }
        while ( v7 != v8 );
        v12 = (__int64)&v5[v10];
      }
    }
    else
    {
      v7 = &v5[*(unsigned int *)(a2 + 24)];
      v8 = sub_16CC9F0(a2, v9);
      if ( v9 == *v8 )
      {
        v5 = *(_QWORD **)(a2 + 16);
        v12 = (__int64)(v5 == *(_QWORD **)(a2 + 8) ? &v5[*(unsigned int *)(a2 + 28)] : &v5[*(unsigned int *)(a2 + 24)]);
      }
      else
      {
        v5 = *(_QWORD **)(a2 + 16);
        if ( v5 != *(_QWORD **)(a2 + 8) )
        {
          v8 = &v5[*(unsigned int *)(a2 + 24)];
          goto LABEL_8;
        }
        v8 = &v5[*(unsigned int *)(a2 + 28)];
        v12 = (__int64)v8;
      }
    }
    while ( (_QWORD *)v12 != v8 && *v8 >= 0xFFFFFFFFFFFFFFFELL )
      ++v8;
LABEL_8:
    if ( v8 != v7 )
      break;
LABEL_11:
    v4 += 3;
    if ( v4 == v13 )
      return 0;
  }
  if ( v6 != 1 )
  {
    v6 = 1;
    goto LABEL_11;
  }
  return 1;
}
