// Function: sub_301FB00
// Address: 0x301fb00
//
__int64 __fastcall sub_301FB00(__int64 a1)
{
  __int64 v1; // rbp
  __int64 *v2; // rdi
  __int64 result; // rax
  const char *v4; // [rsp-38h] [rbp-38h] BYREF
  char v5; // [rsp-18h] [rbp-18h]
  char v6; // [rsp-17h] [rbp-17h]
  __int64 v7; // [rsp-8h] [rbp-8h]

  if ( *(_BYTE *)(a1 + 160) )
  {
    v7 = v1;
    v2 = *(__int64 **)(a1 + 8);
    v6 = 1;
    v4 = "\t}";
    v5 = 3;
    return sub_E99A90(v2, (__int64)&v4);
  }
  return result;
}
