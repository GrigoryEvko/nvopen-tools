// Function: sub_68A000
// Address: 0x68a000
//
__int64 __fastcall sub_68A000(__int64 a1, __int64 a2)
{
  const char *v4; // rdi
  char *v5; // r12
  size_t v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 v10; // rsi
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned __int64 v16; // [rsp+0h] [rbp-60h]

  v16 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
  {
    v4 = *(const char **)(a1 + 24);
    v5 = *(char **)(a1 + 8);
  }
  else
  {
    v5 = *(char **)(a1 + 8);
    v4 = v5;
  }
  v6 = strlen(v4);
  sub_878540(v5, v6);
  v9 = *(_QWORD *)(v16 + 24);
  if ( v9 )
  {
    while ( 1 )
    {
      if ( (*(_DWORD *)(v9 + 80) & 0x400000FF) == 0x4000000B )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 152LL);
        if ( v10 == a2 || (unsigned int)sub_8D97D0(a2, v10, 0, v7, v8) )
          return v9;
      }
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_10;
    }
  }
  else
  {
LABEL_10:
    v12 = (_QWORD *)sub_646F50(a2, 1, 0);
    v13 = v12[14];
    qmemcpy(v12, (const void *)a1, 0x170u);
    v12[19] = a2;
    v12[14] = v13;
    v14 = sub_87EBB0(11, v16);
    *(_BYTE *)(v14 + 83) |= 0x40u;
    v15 = v14;
    *(_QWORD *)(v14 + 88) = v12;
    *v12 = v14;
    sub_885A00(v14, 0, 1);
    return v15;
  }
}
