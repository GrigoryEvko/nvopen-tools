// Function: sub_B46560
// Address: 0xb46560
//
char __fastcall sub_B46560(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // rax
  __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rdi
  unsigned int v6; // ebx
  bool v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ebx

  v1 = *a1;
  if ( (unsigned int)(v1 - 29) <= 0x25 )
  {
    if ( (unsigned int)(v1 - 29) > 0x23 || v1 != 34 && (unsigned __int8)(v1 - 61) <= 1u )
    {
      LOWORD(v2) = *((_WORD *)a1 + 1) & 1;
      return v2;
    }
LABEL_11:
    LOBYTE(v2) = 0;
    return v2;
  }
  if ( v1 != 85 )
    goto LABEL_11;
  v3 = *((_QWORD *)a1 - 4);
  LOBYTE(v2) = 0;
  if ( v3 && !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
  {
    v4 = *(_DWORD *)(v3 + 36);
    if ( (unsigned int)(v4 - 238) <= 7 && ((1LL << ((unsigned __int8)v4 + 18)) & 0xAD) != 0 )
    {
      v5 = *(_QWORD *)&a1[32 * (3LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
      v6 = *(_DWORD *)(v5 + 32);
      if ( v6 <= 0x40 )
        v7 = *(_QWORD *)(v5 + 24) == 0;
      else
        v7 = v6 == (unsigned int)sub_C444A0(v5 + 24);
      LOBYTE(v2) = !v7;
    }
    else
    {
      if ( v4 == 231 )
      {
        v8 = 2;
        v9 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
      }
      else
      {
        LOBYTE(v2) = 0;
        if ( v4 != 232 )
          return v2;
        v8 = 3;
        v9 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
      }
      v2 = *(_QWORD *)&a1[32 * (v8 - v9)];
      v10 = *(_DWORD *)(v2 + 32);
      if ( v10 <= 0x40 )
        LOBYTE(v2) = *(_QWORD *)(v2 + 24) == 1;
      else
        LOBYTE(v2) = v10 - 1 == (unsigned int)sub_C444A0(v2 + 24);
    }
  }
  return v2;
}
