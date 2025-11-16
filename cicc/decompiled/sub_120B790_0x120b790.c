// Function: sub_120B790
// Address: 0x120b790
//
__int64 __fastcall sub_120B790(__int64 a1)
{
  unsigned int v1; // r12d
  _QWORD *v3; // r15
  _BYTE *v4; // r14
  size_t v5; // r13
  _QWORD *v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rdx
  size_t v9; // rcx
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  size_t v12; // rdx
  size_t v13; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v14; // [rsp+10h] [rbp-50h] BYREF
  size_t n; // [rsp+18h] [rbp-48h]
  _QWORD src[8]; // [rsp+20h] [rbp-40h] BYREF

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after source_filename") )
    return 1;
  v1 = sub_120B3D0(a1, a1 + 1752);
  if ( (_BYTE)v1 )
    return 1;
  v3 = *(_QWORD **)(a1 + 344);
  if ( !v3 )
    return v1;
  v4 = *(_BYTE **)(a1 + 1752);
  v5 = *(_QWORD *)(a1 + 1760);
  v14 = src;
  LOBYTE(v1) = v4 == 0 && &v4[v5] != 0;
  if ( (_BYTE)v1 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v13 = v5;
  if ( v5 > 0xF )
  {
    v14 = (_QWORD *)sub_22409D0(&v14, &v13, 0);
    v11 = v14;
    src[0] = v13;
  }
  else
  {
    if ( v5 == 1 )
    {
      LOBYTE(src[0]) = *v4;
      v6 = src;
      goto LABEL_10;
    }
    if ( !v5 )
    {
      v6 = src;
      goto LABEL_10;
    }
    v11 = src;
  }
  memcpy(v11, v4, v5);
  v5 = v13;
  v6 = v14;
LABEL_10:
  n = v5;
  *((_BYTE *)v6 + v5) = 0;
  v7 = (_BYTE *)v3[25];
  if ( v14 == src )
  {
    v12 = n;
    if ( n )
    {
      if ( n == 1 )
        *v7 = src[0];
      else
        memcpy(v7, src, n);
      v12 = n;
      v7 = (_BYTE *)v3[25];
    }
    v3[26] = v12;
    v7[v12] = 0;
    v7 = v14;
    goto LABEL_14;
  }
  v8 = src[0];
  v9 = n;
  if ( v7 == (_BYTE *)(v3 + 27) )
  {
    v3[25] = v14;
    v3[26] = v9;
    v3[27] = v8;
  }
  else
  {
    v10 = v3[27];
    v3[25] = v14;
    v3[26] = v9;
    v3[27] = v8;
    if ( v7 )
    {
      v14 = v7;
      src[0] = v10;
      goto LABEL_14;
    }
  }
  v14 = src;
  v7 = src;
LABEL_14:
  n = 0;
  *v7 = 0;
  if ( v14 != src )
    j_j___libc_free_0(v14, src[0] + 1LL);
  return v1;
}
