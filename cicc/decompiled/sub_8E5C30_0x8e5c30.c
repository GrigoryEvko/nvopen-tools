// Function: sub_8E5C30
// Address: 0x8e5c30
//
unsigned __int8 *__fastcall sub_8E5C30(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int8 *v3; // r13
  bool v4; // zf
  unsigned __int8 *v6; // rax
  unsigned __int8 v7; // al
  __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  char s[96]; // [rsp+10h] [rbp-60h] BYREF

  v2 = 1;
  v3 = (unsigned __int8 *)(a1 + 1);
  v4 = *(_BYTE *)(a1 + 1) == 95;
  v8 = 1;
  if ( v4 )
    goto LABEL_2;
  v6 = sub_8E5810((unsigned __int8 *)(a1 + 1), &v8, a2);
  v3 = v6;
  if ( v8 < 0 )
  {
    if ( !*(_DWORD *)(a2 + 24) )
    {
      ++*(_QWORD *)(a2 + 32);
      ++*(_QWORD *)(a2 + 48);
      *(_DWORD *)(a2 + 24) = 1;
    }
    v8 = 0;
    v2 = 0;
    v7 = *v6;
  }
  else
  {
    v2 = v8 + 2;
    v8 += 2;
    v7 = *v6;
  }
  if ( v7 == 95 )
  {
LABEL_2:
    ++v3;
  }
  else if ( !*(_DWORD *)(a2 + 24) )
  {
    ++*(_QWORD *)(a2 + 32);
    ++*(_QWORD *)(a2 + 48);
    *(_DWORD *)(a2 + 24) = 1;
  }
  sprintf(s, "T%ld", v2);
  if ( !*(_QWORD *)(a2 + 32) )
    sub_8E5790((unsigned __int8 *)s, a2);
  return v3;
}
