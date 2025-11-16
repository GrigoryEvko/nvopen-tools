// Function: sub_9C1440
// Address: 0x9c1440
//
unsigned __int64 __fastcall sub_9C1440(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 *v5; // r13
  __int64 v6; // rax
  unsigned __int64 result; // rax
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_BYTE *)(a2 - 16);
  if ( (v2 & 2) != 0 )
  {
    v3 = *(unsigned int *)(a2 - 24);
    if ( (_DWORD)v3 )
    {
      v4 = *(__int64 **)(a2 - 32);
      goto LABEL_4;
    }
LABEL_7:
    v8[0] = a2;
    return sub_9C0E00(a1, v8);
  }
  LOWORD(v3) = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) == 0 )
    goto LABEL_7;
  v3 = (unsigned __int8)v3;
  v4 = (__int64 *)(a2 - 8LL * ((v2 >> 2) & 0xF) - 16);
LABEL_4:
  v5 = &v4[v3];
  do
  {
    v6 = *v4++;
    v8[0] = v6;
    result = sub_9C0E00(a1, v8);
  }
  while ( v5 != v4 );
  return result;
}
