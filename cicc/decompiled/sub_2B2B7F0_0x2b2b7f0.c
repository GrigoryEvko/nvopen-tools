// Function: sub_2B2B7F0
// Address: 0x2b2b7f0
//
_BYTE *__fastcall sub_2B2B7F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  _BYTE **v3; // r8
  _BYTE *result; // rax
  unsigned int v5; // eax
  __int64 v6; // r9
  _BYTE **v7; // r8
  unsigned int *v8; // r9
  unsigned int v9; // [rsp-24h] [rbp-24h] BYREF
  unsigned int **v10; // [rsp-20h] [rbp-20h] BYREF
  _QWORD v11[3]; // [rsp-18h] [rbp-18h] BYREF

  v3 = *(_BYTE ***)a2;
  if ( (unsigned __int8)(**(_BYTE **)(a2 + 416) - 61) <= 1u
    && *(_DWORD *)(a2 + 104) == 2
    && (v5 = *(_DWORD *)(a2 + 152)) != 0 )
  {
    v11[2] = v2;
    v6 = *(_QWORD *)(a2 + 144);
    v9 = v5;
    v11[0] = v6;
    v11[1] = v5;
    v10 = (unsigned int **)v11;
    if ( sub_2B09970(&v10, &v9) )
    {
      result = v7[*v8];
      if ( *result > 0x1Cu )
        return result;
      return 0;
    }
    result = *v7;
    if ( **v7 <= 0x1Cu )
      return 0;
  }
  else
  {
    result = *v3;
    if ( **v3 <= 0x1Cu )
      return 0;
  }
  return result;
}
