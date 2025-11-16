// Function: sub_1023CA0
// Address: 0x1023ca0
//
__int64 __fastcall sub_1023CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v6; // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  _BYTE *v21; // r8
  __int64 v22; // r14
  _BYTE *i; // r13
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // [rsp+0h] [rbp-90h]
  int v35; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+18h] [rbp-78h]
  _BYTE *v37; // [rsp+20h] [rbp-70h]
  __int64 *v38; // [rsp+28h] [rbp-68h]
  char v39; // [rsp+36h] [rbp-5Ah]
  unsigned __int8 v42; // [rsp+38h] [rbp-58h]
  _BYTE *v43; // [rsp+40h] [rbp-50h] BYREF
  __int64 v44; // [rsp+48h] [rbp-48h]
  _BYTE v45[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = a4;
  v9 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL);
  if ( (_BYTE)v9 != 12 )
  {
    if ( (unsigned __int8)v9 > 0xEu )
      return 0;
    v10 = 16397;
    if ( !_bittest64(&v10, v9) )
      return 0;
    if ( (unsigned __int8)v9 <= 3u || (_BYTE)v9 == 5 || (v9 & 0xFD) == 4 )
      return sub_10235B0(a1, a2, *(_QWORD **)(a3 + 112), a4);
  }
  v12 = sub_DEEF40(a3, a1);
  if ( *(_WORD *)(v12 + 24) == 8 )
    return sub_10238A0(a1, a2, *(_QWORD *)(a3 + 112), v6, (__int64 *)v12, 0);
  if ( !a5 )
    return 0;
  v13 = a1;
  v14 = sub_DEF530(a3, a1);
  v38 = (__int64 *)v14;
  if ( !v14 )
    return 0;
  if ( *(_WORD *)(v12 + 24) != 15 )
  {
LABEL_40:
    v12 = (__int64)v38;
    return sub_10238A0(a1, a2, *(_QWORD *)(a3 + 112), v6, (__int64 *)v12, 0);
  }
  if ( v12 == v14 )
    return sub_10238A0(a1, a2, *(_QWORD *)(a3 + 112), v6, (__int64 *)v12, 0);
  v43 = v45;
  v15 = *(_QWORD *)(v14 + 48);
  v44 = 0x200000000LL;
  v16 = *(_QWORD *)(v12 - 8);
  v37 = (_BYTE *)v16;
  v17 = sub_D47930(v15);
  if ( !v17 )
    goto LABEL_38;
  v18 = *(_QWORD *)(v16 - 8);
  v35 = *(_DWORD *)(v16 + 4);
  v19 = 0x1FFFFFFFE0LL;
  v13 = v35 & 0x7FFFFFF;
  if ( (v35 & 0x7FFFFFF) != 0 )
  {
    v20 = 0;
    do
    {
      if ( v17 == *(_QWORD *)(v18 + 32LL * *(unsigned int *)(v16 + 72) + 8 * v20) )
      {
        v19 = 32 * v20;
        goto LABEL_22;
      }
      ++v20;
    }
    while ( (_DWORD)v13 != (_DWORD)v20 );
    v19 = 0x1FFFFFFFE0LL;
  }
LABEL_22:
  v21 = *(_BYTE **)(v18 + v19);
  if ( !v21 )
    goto LABEL_38;
  v22 = 0;
  v39 = 0;
  if ( *v21 >= 0x1Du )
    v22 = *(_QWORD *)(v18 + v19);
  v36 = v6;
  for ( i = *(_BYTE **)(v18 + v19); ; i = (_BYTE *)v22 )
  {
    if ( v37 == i )
    {
      v6 = v36;
      goto LABEL_55;
    }
    if ( !v22 )
      goto LABEL_37;
    v13 = *(_QWORD *)(v22 + 40);
    if ( *(_BYTE *)(v15 + 84) )
    {
      v24 = *(_QWORD **)(v15 + 64);
      v25 = &v24[*(unsigned int *)(v15 + 76)];
      if ( v24 == v25 )
        goto LABEL_37;
      while ( v13 != *v24 )
      {
        if ( v25 == ++v24 )
          goto LABEL_37;
      }
    }
    else if ( !sub_C8CA60(v15 + 56, v13) )
    {
      goto LABEL_37;
    }
    v13 = sub_DEEF40(a3, (__int64)i);
    if ( *(_WORD *)(v13 + 24) == 8 && sub_DC0560(a3, (_QWORD *)v13, (__int64)v38, v26, v27) || v39 )
    {
      v29 = (unsigned int)v44;
      if ( (_DWORD)v44 )
      {
        v30 = *(_QWORD *)(v22 + 16);
        if ( !v30 || *(_QWORD *)(v30 + 8) )
        {
LABEL_37:
          v6 = v36;
          goto LABEL_38;
        }
      }
      v26 = HIDWORD(v44);
      if ( (unsigned __int64)(unsigned int)v44 + 1 > HIDWORD(v44) )
      {
        v13 = (__int64)v45;
        sub_C8D5F0((__int64)&v43, v45, (unsigned int)v44 + 1LL, 8u, v27, v28);
        v29 = (unsigned int)v44;
      }
      *(_QWORD *)&v43[8 * v29] = v22;
      LODWORD(v44) = v44 + 1;
      v39 = a5;
    }
    if ( (unsigned __int8)(*i - 42) > 0x11u )
      goto LABEL_37;
    v22 = *((_QWORD *)i - 4);
    v13 = *((_QWORD *)i - 8);
    v34 = v13;
    if ( !(unsigned __int8)sub_D48480(v15, v13, v13, v26) )
    {
      v13 = v22;
      if ( !(unsigned __int8)sub_D48480(v15, v22, v31, v32) )
        goto LABEL_37;
      v22 = v34;
    }
    if ( !v22 )
      goto LABEL_37;
    if ( *(_BYTE *)v22 <= 0x1Cu )
      break;
  }
  v6 = v36;
  if ( v37 != (_BYTE *)v22 )
    goto LABEL_38;
LABEL_55:
  if ( !v39 )
  {
LABEL_38:
    if ( v43 != v45 )
      _libc_free(v43, v13);
    goto LABEL_40;
  }
  v33 = a2;
  result = sub_10238A0(a1, a2, *(_QWORD *)(a3 + 112), v6, v38, (__int64)&v43);
  if ( v43 != v45 )
  {
    v42 = result;
    _libc_free(v43, v33);
    return v42;
  }
  return result;
}
