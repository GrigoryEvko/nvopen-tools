// Function: sub_11FF280
// Address: 0x11ff280
//
__int64 __fastcall sub_11FF280(__int64 a1)
{
  __int64 result; // rax
  size_t v2; // rdx
  unsigned __int64 v3; // rsi
  const char *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  result = sub_11FE410(a1, 0x200u);
  if ( (unsigned int)result > 1 && **(_BYTE **)a1 == 58 )
  {
    ++*(_QWORD *)a1;
    v2 = *(_QWORD *)(a1 + 80);
    if ( v2 && memchr(*(const void **)(a1 + 72), 0, v2) )
    {
      v3 = *(_QWORD *)(a1 + 56);
      v6 = 1;
      v5 = 3;
      v4 = "NUL character is not allowed in names";
      sub_11FD800(a1, v3, (__int64)&v4, 2);
      return 1;
    }
    else
    {
      return 507;
    }
  }
  return result;
}
