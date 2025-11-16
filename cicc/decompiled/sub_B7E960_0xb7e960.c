// Function: sub_B7E960
// Address: 0xb7e960
//
__int64 __fastcall sub_B7E960(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rbp
  __int64 result; // rax
  _UNKNOWN **v5; // [rsp-18h] [rbp-18h] BYREF
  int v6; // [rsp-10h] [rbp-10h]
  char v7; // [rsp-Ch] [rbp-Ch]
  __int64 v8; // [rsp-8h] [rbp-8h]

  result = *(unsigned int *)(a1 + 136);
  if ( a3 || !*(_BYTE *)(a1 + 156) || *(_DWORD *)(a1 + 152) != (_DWORD)result )
  {
    v8 = v3;
    v5 = &off_49DA5D8;
    v7 = 1;
    v6 = result;
    return ((__int64 (__fastcall *)(__int64, __int64, _UNKNOWN ***, __int64, __int64))sub_C55D10)(
             a1 + 160,
             a1,
             &v5,
             a1 + 144,
             a2);
  }
  return result;
}
