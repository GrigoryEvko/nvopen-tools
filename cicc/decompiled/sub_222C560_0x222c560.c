// Function: sub_222C560
// Address: 0x222c560
//
char __fastcall sub_222C560(__int64 a1, __int64 a2)
{
  __int64 v3; // rbp
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // r12
  const void *v9; // rsi
  __int64 v10; // r12
  size_t v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax

  if ( (unsigned __int8)sub_2231390(a2, a2) )
    v3 = sub_222FD10(a2);
  else
    v3 = 0;
  LOBYTE(v4) = sub_2207CD0((_QWORD *)(a1 + 104));
  if ( !(_BYTE)v4 )
    goto LABEL_13;
  if ( !*(_BYTE *)(a1 + 169) && !*(_BYTE *)(a1 + 170) )
  {
    *(_QWORD *)(a1 + 200) = v3;
    return v4;
  }
  v7 = *(_QWORD *)(a1 + 200);
  if ( !v7 )
    goto LABEL_24;
  LODWORD(v4) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7);
  if ( (_DWORD)v4 == -1 )
    goto LABEL_18;
  if ( !*(_BYTE *)(a1 + 169) )
  {
    if ( !*(_BYTE *)(a1 + 170) )
      goto LABEL_13;
    LOBYTE(v4) = sub_222BE90(a1, a2, v5, v6);
    if ( (_BYTE)v4 )
    {
      v4 = *(_QWORD *)(a1 + 152);
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 8) = v4;
      *(_QWORD *)(a1 + 16) = v4;
      *(_QWORD *)(a1 + 24) = v4;
      *(_QWORD *)(a1 + 48) = 0;
      goto LABEL_13;
    }
LABEL_18:
    *(_QWORD *)(a1 + 200) = 0;
    return v4;
  }
  v7 = *(_QWORD *)(a1 + 200);
  if ( !v7 )
LABEL_24:
    sub_426219(v7, a2, v5, v6);
  LOBYTE(v4) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 48LL))(v7);
  if ( !(_BYTE)v4 )
  {
    v8 = *(_QWORD *)(a1 + 208);
    v9 = (const void *)(v8
                      + (*(int (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 200) + 56LL))(
                          *(_QWORD *)(a1 + 200),
                          a1 + 140,
                          v8,
                          *(_QWORD *)(a1 + 224),
                          *(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)));
    v10 = *(_QWORD *)(a1 + 232);
    *(_QWORD *)(a1 + 224) = v9;
    v11 = v10 - (_QWORD)v9;
    if ( v11 )
      memmove(*(void **)(a1 + 208), v9, v11);
    v12 = *(_QWORD *)(a1 + 208);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 224) = v12;
    *(_QWORD *)(a1 + 232) = v11 + v12;
    v13 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = v13;
    *(_QWORD *)(a1 + 16) = v13;
    *(_QWORD *)(a1 + 24) = v13;
    v4 = *(_QWORD *)(a1 + 124);
    *(_QWORD *)(a1 + 132) = v4;
    *(_QWORD *)(a1 + 140) = v4;
    goto LABEL_13;
  }
  if ( !v3 )
    goto LABEL_18;
  LOBYTE(v4) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 48LL))(v3);
  if ( !(_BYTE)v4 )
  {
    v4 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a1 + 32LL))(
           a1,
           0,
           1,
           *(unsigned int *)(a1 + 120));
    if ( v4 == -1 )
      goto LABEL_18;
  }
LABEL_13:
  *(_QWORD *)(a1 + 200) = v3;
  return v4;
}
