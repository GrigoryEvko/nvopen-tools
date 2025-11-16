// Function: sub_388C5A0
// Address: 0x388c5a0
//
__int64 __fastcall sub_388C5A0(__int64 a1, unsigned int *a2)
{
  __int64 result; // rax
  int v3; // eax
  unsigned __int64 v4; // r13
  unsigned int v5; // eax
  const char *v6; // rax
  const char *v7; // [rsp-48h] [rbp-48h] BYREF
  char v8; // [rsp-38h] [rbp-38h]
  char v9; // [rsp-37h] [rbp-37h]

  *a2 = 0;
  if ( *(_DWORD *)(a1 + 64) != 88 )
    return 0;
  v3 = sub_3887100(a1 + 8);
  v4 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 64) = v3;
  result = sub_388BA90(a1, a2);
  if ( !(_BYTE)result )
  {
    v5 = *a2;
    if ( !*a2 || (v5 & (v5 - 1)) != 0 )
    {
      v9 = 1;
      v6 = "alignment is not a power of two";
    }
    else
    {
      if ( v5 <= 0x20000000 )
        return 0;
      v9 = 1;
      v6 = "huge alignments are not supported yet";
    }
    v7 = v6;
    v8 = 3;
    return sub_38814C0(a1 + 8, v4, (__int64)&v7);
  }
  return result;
}
