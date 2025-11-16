// Function: sub_DB6680
// Address: 0xdb6680
//
__int64 __fastcall sub_DB6680(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  __int64 v7; // r8
  __int64 result; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 v18; // r9
  __int64 **v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // [rsp+18h] [rbp-128h]
  unsigned __int8 v24; // [rsp+18h] [rbp-128h]
  unsigned __int8 v25; // [rsp+18h] [rbp-128h]
  _QWORD *v26; // [rsp+20h] [rbp-120h] BYREF
  __int64 v27; // [rsp+28h] [rbp-118h]
  _QWORD v28[8]; // [rsp+30h] [rbp-110h] BYREF
  __int64 v29; // [rsp+70h] [rbp-D0h] BYREF
  __int64 *v30; // [rsp+78h] [rbp-C8h]
  __int64 v31; // [rsp+80h] [rbp-C0h]
  int v32; // [rsp+88h] [rbp-B8h]
  char v33; // [rsp+8Ch] [rbp-B4h]
  __int64 v34; // [rsp+90h] [rbp-B0h] BYREF

  v23 = sub_D46F00(a3);
  if ( !v23 )
    return 0;
  v6 = (__int64 *)a3;
  if ( !(unsigned __int8)sub_DB5FD0(a1, a3) )
    return 0;
  v32 = 0;
  v9 = v28;
  v10 = (__int64)&v26;
  v30 = &v34;
  v26 = v28;
  v33 = 1;
  v34 = a2;
  v29 = 1;
  v28[0] = a2;
  v31 = 0x100000010LL;
  v27 = 0x800000001LL;
  LODWORD(result) = 1;
  while ( 1 )
  {
    v11 = (unsigned int)result;
    result = (unsigned int)(result - 1);
    v12 = v9[v11 - 1];
    LODWORD(v27) = result;
    v13 = *(_QWORD *)(v12 + 16);
    if ( v13 )
      break;
LABEL_27:
    if ( !(_DWORD)result )
    {
      if ( v9 != v28 )
        goto LABEL_29;
      goto LABEL_30;
    }
  }
  while ( 1 )
  {
    v14 = *(_QWORD *)(v13 + 24);
    v6 = &v29;
    if ( (unsigned __int8)sub_98D5C0((unsigned __int8 *)v14, (__int64)&v29, (unsigned __int8 *)v12, v10, v7) )
    {
      v6 = *(__int64 **)(v14 + 40);
      result = sub_B19720(*(_QWORD *)(a1 + 40), (__int64)v6, v23);
      if ( (_BYTE)result )
        break;
    }
    if ( !(unsigned __int8)sub_98D0D0(v13, (__int64)v6, v15, v16, v17) )
      goto LABEL_8;
    v6 = *(__int64 **)(v14 + 40);
    if ( *(_BYTE *)(a3 + 84) )
    {
      v19 = *(__int64 ***)(a3 + 64);
      v12 = (__int64)&v19[*(unsigned int *)(a3 + 76)];
      if ( v19 == (__int64 **)v12 )
        goto LABEL_8;
      while ( v6 != *v19 )
      {
        if ( (__int64 **)v12 == ++v19 )
          goto LABEL_8;
      }
    }
    else if ( !sub_C8CA60(a3 + 56, (__int64)v6) )
    {
      goto LABEL_8;
    }
    if ( !v33 )
      goto LABEL_32;
    v20 = v30;
    v10 = HIDWORD(v31);
    v12 = (__int64)&v30[HIDWORD(v31)];
    if ( v30 != (__int64 *)v12 )
    {
      while ( v14 != *v20 )
      {
        if ( (__int64 *)v12 == ++v20 )
          goto LABEL_21;
      }
      goto LABEL_8;
    }
LABEL_21:
    if ( HIDWORD(v31) < (unsigned int)v31 )
    {
      ++HIDWORD(v31);
      *(_QWORD *)v12 = v14;
      ++v29;
LABEL_23:
      v21 = (unsigned int)v27;
      v10 = HIDWORD(v27);
      v22 = (unsigned int)v27 + 1LL;
      if ( v22 > HIDWORD(v27) )
      {
        v6 = v28;
        sub_C8D5F0((__int64)&v26, v28, v22, 8u, v7, v18);
        v21 = (unsigned int)v27;
      }
      v12 = (__int64)v26;
      v26[v21] = v14;
      LODWORD(v27) = v27 + 1;
      v13 = *(_QWORD *)(v13 + 8);
      if ( !v13 )
      {
LABEL_26:
        result = (unsigned int)v27;
        v9 = v26;
        goto LABEL_27;
      }
    }
    else
    {
LABEL_32:
      v6 = (__int64 *)v14;
      sub_C8CC70((__int64)&v29, v14, v12, v10, v7, v18);
      if ( (_BYTE)v12 )
        goto LABEL_23;
LABEL_8:
      v13 = *(_QWORD *)(v13 + 8);
      if ( !v13 )
        goto LABEL_26;
    }
  }
  v9 = v26;
  if ( v26 == v28 )
    goto LABEL_30;
LABEL_29:
  v24 = result;
  _libc_free(v9, v6);
  result = v24;
LABEL_30:
  if ( !v33 )
  {
    v25 = result;
    _libc_free(v30, v6);
    return v25;
  }
  return result;
}
