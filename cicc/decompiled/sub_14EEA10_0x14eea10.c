// Function: sub_14EEA10
// Address: 0x14eea10
//
__int64 __fastcall sub_14EEA10(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rdx
  char v5; // al
  const char *v7; // rax
  __int64 v8; // rax
  __int64 v9; // [rsp+8h] [rbp-38h] BYREF
  const char *v10; // [rsp+10h] [rbp-30h] BYREF
  char v11; // [rsp+20h] [rbp-20h]
  char v12; // [rsp+21h] [rbp-1Fh]

  if ( a4 )
  {
    v4 = *a3;
    if ( (unsigned int)v4 <= 2 )
    {
      *(_BYTE *)(a2 + 384) = (_DWORD)v4 == 2;
      v5 = *(_BYTE *)(a1 + 8);
      *(_DWORD *)a1 = v4;
      *(_BYTE *)(a1 + 8) = v5 & 0xFC | 2;
      return a1;
    }
    v12 = 1;
    v7 = "Invalid value";
  }
  else
  {
    v12 = 1;
    v7 = "Invalid record";
  }
  v10 = v7;
  v11 = 3;
  sub_14EE4B0(&v9, a2, (__int64)&v10);
  v8 = v9;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v8 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
