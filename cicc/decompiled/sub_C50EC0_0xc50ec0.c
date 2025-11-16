// Function: sub_C50EC0
// Address: 0xc50ec0
//
__int64 __fastcall sub_C50EC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 result; // rax

  if ( !a1 )
    return result;
  if ( *(_BYTE *)(a1 + 308) )
  {
    if ( *(_BYTE *)(a1 + 148) )
      goto LABEL_4;
LABEL_12:
    _libc_free(*(_QWORD *)(a1 + 128), a2);
    v3 = *(_QWORD *)(a1 + 72);
    if ( v3 == a1 + 88 )
      goto LABEL_6;
    goto LABEL_5;
  }
  _libc_free(*(_QWORD *)(a1 + 288), a2);
  if ( !*(_BYTE *)(a1 + 148) )
    goto LABEL_12;
LABEL_4:
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != a1 + 88 )
LABEL_5:
    _libc_free(v3, a2);
LABEL_6:
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 64) - v4);
  if ( *(_QWORD *)a1 != a1 + 16 )
    j_j___libc_free_0(*(_QWORD *)a1, *(_QWORD *)(a1 + 16) + 1LL);
  return j_j___libc_free_0(a1, 352);
}
