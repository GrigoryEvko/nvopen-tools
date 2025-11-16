// Function: sub_222B830
// Address: 0x222b830
//
__int64 __fastcall sub_222B830(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  char *v4; // r13
  signed __int64 v5; // r12
  _BYTE *v7; // rax
  __int64 v8; // r14
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // ebx
  char v14; // al
  __int64 v16; // rax
  const void *v17; // rsi
  size_t v18; // rbx
  ssize_t v19; // rax
  void *v20; // rdi
  __int64 v21; // rax
  int *v22; // rax

  v4 = (char *)a2;
  v5 = (signed __int64)a3;
  if ( *(_BYTE *)(a1 + 192) )
  {
    v7 = *(_BYTE **)(a1 + 16);
    a3 = *(_BYTE **)(a1 + 8);
    v8 = 0;
    if ( v5 <= 0 || v7 != a3 )
      goto LABEL_4;
    --v5;
    a4 = a2 + 1;
    *(_BYTE *)a2 = *v7;
    v7 = (_BYTE *)(*(_QWORD *)(a1 + 16) + 1LL);
    v9 = *(_BYTE *)(a1 + 192) == 0;
    *(_QWORD *)(a1 + 16) = v7;
    if ( !v9 )
    {
      a3 = *(_BYTE **)(a1 + 8);
      v4 = (char *)(a2 + 1);
      v8 = 1;
LABEL_4:
      v9 = a3 == v7;
      a4 = *(_QWORD *)(a1 + 152);
      a3 = *(_BYTE **)(a1 + 184);
      *(_BYTE *)(a1 + 192) = 0;
      v10 = *(_QWORD *)(a1 + 176) + !v9;
      *(_QWORD *)(a1 + 8) = a4;
      *(_QWORD *)(a1 + 176) = v10;
      *(_QWORD *)(a1 + 16) = v10;
      *(_QWORD *)(a1 + 24) = a3;
      goto LABEL_5;
    }
    v4 = (char *)(a2 + 1);
    v8 = 1;
  }
  else if ( *(_BYTE *)(a1 + 170) )
  {
    a2 = 0xFFFFFFFFLL;
    if ( (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 104LL))(a1, 0xFFFFFFFFLL) == -1 )
      return 0;
    v16 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(a1 + 40) = 0;
    v8 = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 8) = v16;
    *(_QWORD *)(a1 + 16) = v16;
    *(_QWORD *)(a1 + 24) = v16;
    *(_QWORD *)(a1 + 48) = 0;
    *(_BYTE *)(a1 + 170) = 0;
  }
  else
  {
    v8 = 0;
  }
LABEL_5:
  v11 = 2;
  if ( *(_QWORD *)(a1 + 160) >= 2u )
    v11 = *(_QWORD *)(a1 + 160);
  if ( v5 <= v11 - 1 )
    goto LABEL_11;
  v12 = *(_QWORD *)(a1 + 200);
  if ( !v12 )
    sub_426219(0, a2, a3, a4);
  v13 = *(_DWORD *)(a1 + 120);
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 48LL))(v12);
  if ( (v13 & 8) != 0 && v14 )
  {
    v17 = *(const void **)(a1 + 16);
    v18 = *(_QWORD *)(a1 + 24) - (_QWORD)v17;
    if ( v18 )
    {
      v20 = v4;
      v4 += v18;
      v8 += v18;
      memcpy(v20, v17, v18);
      v5 -= v18;
      *(_QWORD *)(a1 + 16) += v18;
    }
    while ( 1 )
    {
      v19 = sub_2207DA0((FILE **)(a1 + 104), v4, v5);
      if ( v19 == -1 )
      {
        v22 = __errno_location();
        sub_426AAD((__int64)"basic_filebuf::xsgetn error reading the file", *v22);
      }
      if ( !v19 )
        break;
      v8 += v19;
      v5 -= v19;
      if ( !v5 )
        goto LABEL_28;
      v4 += v19;
    }
    if ( !v5 )
    {
LABEL_28:
      *(_BYTE *)(a1 + 169) = 1;
      return v8;
    }
    v21 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 8) = v21;
    *(_QWORD *)(a1 + 16) = v21;
    *(_QWORD *)(a1 + 24) = v21;
    *(_QWORD *)(a1 + 48) = 0;
    *(_BYTE *)(a1 + 169) = 0;
  }
  else
  {
LABEL_11:
    v8 += sub_22406F0(a1, v4, v5);
  }
  return v8;
}
