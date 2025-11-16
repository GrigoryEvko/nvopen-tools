// Function: sub_1B47110
// Address: 0x1b47110
//
__int64 __fastcall sub_1B47110(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4, __int64 **a5, int a6)
{
  __int64 v6; // r14
  __int64 v7; // rdi
  unsigned __int64 v11; // rax
  unsigned int v12; // r11d
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // r9d
  unsigned int v18; // eax
  unsigned int v19; // r8d
  __int64 v20; // rax
  __int64 **v21; // r8
  unsigned int v22; // ebx
  _QWORD *v23; // r14
  __int64 *v24; // rax
  __int64 *v25; // rsi
  unsigned int v26; // edi
  __int64 *v27; // rcx
  __int64 v28; // [rsp+0h] [rbp-50h]
  _BYTE v29[5]; // [rsp+Fh] [rbp-41h]
  _QWORD *v30; // [rsp+10h] [rbp-40h]
  __int64 **v32; // [rsp+18h] [rbp-38h]
  unsigned __int8 v33; // [rsp+18h] [rbp-38h]

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 40);
  if ( v7 == a2 )
    return 0;
  v11 = sub_157EBA0(v7);
  if ( *(_BYTE *)(v11 + 16) != 26 || (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) == 3 || a2 != *(_QWORD *)(v11 - 24) )
    return 1;
  if ( !a3 || (unsigned int)(*(_DWORD *)(a3 + 28) - *(_DWORD *)(a3 + 32)) > 3 )
    return 0;
  *(_DWORD *)&v29[1] = sub_13A0E30(a3, v6);
  if ( *(_DWORD *)&v29[1] )
    return 1;
  *(_DWORD *)v29 = (unsigned __int8)sub_14AF470(v6, 0, 0, 0);
  if ( !v29[0] )
    return 0;
  v18 = sub_1B44750(v6, a5, v14, v15, v16, v17);
  if ( v18 <= *a4 )
  {
    v19 = *a4 - v18;
    goto LABEL_14;
  }
  if ( !byte_4FB7140 || *(_DWORD *)(a3 + 28) != *(_DWORD *)(a3 + 32) || a6 )
    return 0;
  v19 = *(_DWORD *)&v29[1];
LABEL_14:
  *a4 = v19;
  v20 = sub_13CF970(v6);
  v12 = v29[0];
  v30 = (_QWORD *)(v20 + 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
  if ( (_QWORD *)v20 != v30 )
  {
    v21 = a5;
    v28 = v6;
    v22 = a6 + 1;
    v23 = (_QWORD *)v20;
    while ( 1 )
    {
      v32 = v21;
      if ( !(unsigned __int8)sub_1B47330(*v23, a2, a3, a4, v21, v22) )
        return 0;
      v23 += 3;
      v21 = v32;
      if ( v30 == v23 )
      {
        v12 = v29[0];
        v6 = v28;
        break;
      }
    }
  }
  v24 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v24 )
  {
LABEL_20:
    v33 = v12;
    sub_16CCBA0(a3, v6);
    return v33;
  }
  v25 = &v24[*(unsigned int *)(a3 + 28)];
  v26 = *(_DWORD *)(a3 + 28);
  if ( v24 == v25 )
  {
LABEL_33:
    if ( v26 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v26 + 1;
      *v25 = v6;
      ++*(_QWORD *)a3;
      return v12;
    }
    goto LABEL_20;
  }
  v27 = 0;
  while ( v6 != *v24 )
  {
    if ( *v24 == -2 )
      v27 = v24;
    if ( v25 == ++v24 )
    {
      if ( !v27 )
        goto LABEL_33;
      *v27 = v6;
      --*(_DWORD *)(a3 + 32);
      ++*(_QWORD *)a3;
      return v12;
    }
  }
  return v12;
}
