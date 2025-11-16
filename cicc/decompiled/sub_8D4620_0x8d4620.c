// Function: sub_8D4620
// Address: 0x8d4620
//
__int64 __fastcall sub_8D4620(__int64 a1)
{
  __int64 v1; // rbp
  __int64 v2; // rcx
  char i; // al
  __int64 result; // rax
  __int64 v5; // rdi
  int v6; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v7; // [rsp-8h] [rbp-8h]

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v2 = *(_QWORD *)(a1 + 160);
  for ( i = *(_BYTE *)(v2 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  if ( *(_BYTE *)(a1 + 177) != 1 || i != 2 || (*(_BYTE *)(v2 + 162) & 4) == 0 )
    return *(_QWORD *)(a1 + 128) / *(_QWORD *)(v2 + 128);
  v5 = *(_QWORD *)(a1 + 168);
  result = 1;
  if ( *(_BYTE *)(v5 + 173) == 1 )
  {
    v7 = v1;
    v6 = 0;
    return sub_620FA0(v5, &v6);
  }
  return result;
}
