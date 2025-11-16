// Function: sub_EB5D60
// Address: 0xeb5d60
//
__int64 __fastcall sub_EB5D60(__int64 a1, char a2)
{
  char *v3; // rsi
  bool v4; // zf
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r13

  v3 = *(char **)(a1 + 328);
  if ( v3 == *(char **)(a1 + 336) )
  {
    sub_EA9230((char **)(a1 + 320), v3, (_QWORD *)(a1 + 308));
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *(_QWORD *)(a1 + 308);
      v3 = *(char **)(a1 + 328);
    }
    *(_QWORD *)(a1 + 328) = v3 + 8;
  }
  v4 = *(_BYTE *)(a1 + 313) == 0;
  *(_DWORD *)(a1 + 308) = 1;
  if ( v4 )
  {
    sub_EABDC0(a1);
    v7 = v6;
    result = sub_ECE000(a1);
    if ( !(_BYTE)result )
    {
      *(_BYTE *)(a1 + 312) = a2 == (v7 == 0);
      *(_BYTE *)(a1 + 313) = a2 ^ (v7 == 0);
    }
  }
  else
  {
    sub_EB4E00(a1);
    return 0;
  }
  return result;
}
