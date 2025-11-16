// Function: sub_214E300
// Address: 0x214e300
//
_BYTE *__fastcall sub_214E300(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *result; // rax
  __int64 v5; // rdi
  _BYTE *v6; // rax
  void *v7; // rdx
  __int64 v8; // rdi
  char *v9[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+10h] [rbp-40h] BYREF
  char v11; // [rsp+20h] [rbp-30h]

  sub_3937CA0(v9, a2, 1);
  if ( v11 )
  {
    v5 = sub_16E7EE0(a3, v9[0], (size_t)v9[1]);
    v6 = *(_BYTE **)(v5 + 24);
    if ( *(_BYTE **)(v5 + 16) == v6 )
    {
      sub_16E7EE0(v5, "\n", 1u);
    }
    else
    {
      *v6 = 10;
      ++*(_QWORD *)(v5 + 24);
    }
    if ( v11 && (__int64 *)v9[0] != &v10 )
      j_j___libc_free_0(v9[0], v10 + 1);
  }
  result = (_BYTE *)sub_1C2EEA0((__int64)v9, a2);
  if ( BYTE4(v9[0]) )
  {
    v7 = *(void **)(a3 + 24);
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v7 <= 0xEu )
    {
      a3 = sub_16E7EE0(a3, ".local_maxnreg ", 0xFu);
    }
    else
    {
      qmemcpy(v7, ".local_maxnreg ", 15);
      *(_QWORD *)(a3 + 24) += 15LL;
    }
    v8 = sub_16E7A90(a3, LODWORD(v9[0]));
    result = *(_BYTE **)(v8 + 24);
    if ( *(_BYTE **)(v8 + 16) == result )
    {
      return (_BYTE *)sub_16E7EE0(v8, "\n", 1u);
    }
    else
    {
      *result = 10;
      ++*(_QWORD *)(v8 + 24);
    }
  }
  return result;
}
