// Function: sub_1C76EB0
// Address: 0x1c76eb0
//
_QWORD *__fastcall sub_1C76EB0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  _BYTE *v11; // rsi
  __int64 *v12; // r8
  __int64 *v13; // r15
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // r11
  __int64 v20; // rsi
  _QWORD *v21; // rax
  __int64 *v22; // r15
  __int64 *i; // r13
  __int64 v24; // rdi
  int v26; // eax
  int v27; // r10d
  _BYTE *v28; // rsi
  __int64 *v29; // [rsp+0h] [rbp-50h]
  _QWORD v31[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = sub_194ACF0(a4);
  v31[0] = v9;
  v10 = v9;
  if ( a2 )
  {
    *v9 = a2;
    v11 = *(_BYTE **)(a2 + 16);
    if ( v11 == *(_BYTE **)(a2 + 24) )
    {
      sub_13FD960(a2 + 8, v11, v31);
    }
    else
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = v31[0];
        v11 = *(_BYTE **)(a2 + 16);
      }
      *(_QWORD *)(a2 + 16) = v11 + 8;
    }
  }
  else
  {
    v28 = *(_BYTE **)(a4 + 40);
    if ( v28 == *(_BYTE **)(a4 + 48) )
    {
      sub_13FD960(a4 + 32, v28, v31);
    }
    else
    {
      if ( v28 )
      {
        *(_QWORD *)v28 = v9;
        v28 = *(_BYTE **)(a4 + 40);
      }
      *(_QWORD *)(a4 + 40) = v28 + 8;
    }
  }
  if ( a5 )
    sub_14070E0(a5, v10);
  v12 = (__int64 *)a1[5];
  v13 = (__int64 *)a1[4];
  if ( v13 != v12 )
  {
    while ( 1 )
    {
      v14 = *(_DWORD *)(a4 + 24);
      if ( !v14 )
        goto LABEL_10;
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a4 + 8);
      v17 = v15 & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( *v13 != *v18 )
      {
        v26 = 1;
        while ( v19 != -8 )
        {
          v27 = v26 + 1;
          v17 = v15 & (v26 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( *v13 == *v18 )
            goto LABEL_13;
          v26 = v27;
        }
        goto LABEL_10;
      }
LABEL_13:
      if ( a1 == (_QWORD *)v18[1] )
      {
        v20 = *v13;
        v29 = v12;
        ++v13;
        v21 = sub_1C76B50(a3, v20);
        sub_1400330((__int64)v10, v21[2], a4);
        v12 = v29;
        if ( v29 == v13 )
          break;
      }
      else
      {
LABEL_10:
        if ( v12 == ++v13 )
          break;
      }
    }
  }
  v22 = (__int64 *)a1[2];
  for ( i = (__int64 *)a1[1]; v22 != i; ++i )
  {
    v24 = *i;
    sub_1C76EB0(v24, v10, a3, a4, a5);
  }
  return v10;
}
