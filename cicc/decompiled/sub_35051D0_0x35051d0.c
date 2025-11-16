// Function: sub_35051D0
// Address: 0x35051d0
//
_QWORD *__fastcall sub_35051D0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int8 v3; // al
  _BYTE **v4; // rdx
  _QWORD *result; // rax
  __int64 v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = a2 - 16;
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_BYTE ***)(a2 - 32);
  else
    v4 = (_BYTE **)(v2 - 8LL * ((v3 >> 2) & 0xF));
  result = *v4;
  if ( !*v4 )
    return result;
  v6 = (__int64)sub_AF3520(*v4);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
  {
    if ( *(_DWORD *)(a2 - 24) != 2 )
      goto LABEL_6;
    v8 = *(_QWORD *)(a2 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
    {
LABEL_6:
      v12[0] = v6;
      result = sub_32239E0(a1 + 1, v12);
      if ( result )
        result += 2;
      return result;
    }
    v8 = v2 - 8LL * ((v7 >> 2) & 0xF);
  }
  v9 = *(_QWORD *)(v8 + 8);
  if ( !v9 )
    goto LABEL_6;
  v12[0] = v6;
  v12[1] = v9;
  v10 = sub_3505160(a1 + 8, (unsigned __int64)(v9 + 31 * v6) % a1[9], v12, v9 + 31 * v6);
  if ( v10 && (v11 = *v10) != 0 )
    return (_QWORD *)(v11 + 24);
  else
    return 0;
}
