// Function: sub_222B020
// Address: 0x222b020
//
__int64 __fastcall sub_222B020(__int64 a1, unsigned int a2)
{
  char v2; // r12
  unsigned __int64 v3; // rax
  unsigned int v4; // edx
  __int64 result; // rax
  _BYTE *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  if ( (*(_BYTE *)(a1 + 120) & 8) == 0 )
    return 0xFFFFFFFFLL;
  if ( *(_BYTE *)(a1 + 170) )
  {
    if ( (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 104LL))(a1, 0xFFFFFFFFLL) == -1 )
      return 0xFFFFFFFFLL;
    v8 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    v2 = *(_BYTE *)(a1 + 192);
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = v8;
    *(_QWORD *)(a1 + 24) = v8;
    *(_QWORD *)(a1 + 48) = 0;
    *(_BYTE *)(a1 + 170) = 0;
  }
  else
  {
    v2 = *(_BYTE *)(a1 + 192);
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 > *(_QWORD *)(a1 + 8) )
    {
      *(_QWORD *)(a1 + 16) = v3 - 1;
      v4 = *(unsigned __int8 *)(v3 - 1);
      goto LABEL_5;
    }
  }
  if ( (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 32LL))(a1, -1, 1, 24) == -1 )
    return 0xFFFFFFFFLL;
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL))(a1);
  if ( v4 == -1 )
    return 0xFFFFFFFFLL;
LABEL_5:
  result = 0;
  if ( a2 == -1 )
    return result;
  result = v4;
  if ( v4 == a2 )
    return result;
  if ( v2 )
    return 0xFFFFFFFFLL;
  v6 = *(_BYTE **)(a1 + 16);
  if ( !*(_BYTE *)(a1 + 192) )
  {
    *(_QWORD *)(a1 + 176) = v6;
    v7 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 24) = a1 + 172;
    *(_QWORD *)(a1 + 184) = v7;
    v6 = (_BYTE *)(a1 + 171);
    *(_QWORD *)(a1 + 8) = a1 + 171;
    *(_QWORD *)(a1 + 16) = a1 + 171;
    *(_BYTE *)(a1 + 192) = 1;
  }
  *(_BYTE *)(a1 + 169) = 1;
  *v6 = a2;
  return a2;
}
