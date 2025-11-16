// Function: sub_C33D00
// Address: 0xc33d00
//
bool __fastcall sub_C33D00(__int64 a1, __int64 a2)
{
  bool result; // al
  unsigned __int8 v3; // dl
  char v4; // dl
  const void *v5; // r13
  __int64 v6; // rbx
  _BYTE *v7; // rbx
  _BYTE *v8; // rdi

  if ( a1 == a2 )
    return 1;
  result = 0;
  if ( *(_QWORD *)a1 == *(_QWORD *)a2 )
  {
    v3 = *(_BYTE *)(a1 + 20);
    if ( ((v3 ^ *(_BYTE *)(a2 + 20)) & 0xF) == 0 )
    {
      v4 = v3 & 7;
      result = v4 == 0 || v4 == 3;
      if ( result )
        return 1;
      if ( v4 == 1 || *(_DWORD *)(a1 + 16) == *(_DWORD *)(a2 + 16) )
      {
        v5 = (const void *)sub_C33930(a2);
        v6 = sub_C33930(a1);
        v7 = (_BYTE *)(v6 + 8LL * (unsigned int)sub_C337D0(a1));
        v8 = (_BYTE *)sub_C33930(a1);
        if ( v7 != v8 )
          return memcmp(v8, v5, v7 - v8) == 0;
        return 1;
      }
    }
  }
  return result;
}
