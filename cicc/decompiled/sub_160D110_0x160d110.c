// Function: sub_160D110
// Address: 0x160d110
//
__int64 __fastcall sub_160D110(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rbp
  __int64 result; // rax
  _UNKNOWN **v5; // [rsp-18h] [rbp-18h] BYREF
  int v6; // [rsp-10h] [rbp-10h]
  char v7; // [rsp-Ch] [rbp-Ch]
  __int64 v8; // [rsp-8h] [rbp-8h]

  result = *(unsigned int *)(a1 + 160);
  if ( a3 || *(_BYTE *)(a1 + 180) && *(_DWORD *)(a1 + 176) != (_DWORD)result )
  {
    v8 = v3;
    v5 = &off_49ED560;
    v7 = 1;
    v6 = result;
    return ((__int64 (__fastcall *)(__int64, __int64, _UNKNOWN ***, __int64, __int64))sub_16B38E0)(
             a1 + 184,
             a1,
             &v5,
             a1 + 168,
             a2);
  }
  return result;
}
