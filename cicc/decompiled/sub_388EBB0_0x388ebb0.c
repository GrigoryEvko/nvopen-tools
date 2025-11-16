// Function: sub_388EBB0
// Address: 0x388ebb0
//
__int64 __fastcall sub_388EBB0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  unsigned int v3; // r15d
  unsigned int v4; // eax
  int v5; // r8d
  int v6; // r9d
  int v7; // edx
  __int64 v8; // rax
  int v9; // ecx
  unsigned int v10; // eax
  const char *v11; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // [rsp+10h] [rbp-60h]
  int v15; // [rsp+1Ch] [rbp-54h]
  _QWORD v16[2]; // [rsp+20h] [rbp-50h] BYREF
  char v17; // [rsp+30h] [rbp-40h]
  char v18; // [rsp+31h] [rbp-3Fh]

  v14 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_388AF10(a1, 8, "expected '{' here") )
  {
    return 1;
  }
  else if ( *(_DWORD *)(a1 + 64) == 9 )
  {
    v13 = *(_QWORD *)(a1 + 56);
    v18 = 1;
    v17 = 3;
    v16[0] = "expected non-empty list of uselistorder indexes";
    return (unsigned int)sub_38814C0(a1 + 8, v13, (__int64)v16);
  }
  else
  {
    v15 = 0;
    v2 = 0;
    v3 = 1;
    while ( 1 )
    {
      v4 = sub_388BA90(a1, v16);
      if ( (_BYTE)v4 )
        break;
      v7 = v16[0];
      v8 = *(unsigned int *)(a2 + 8);
      v9 = LODWORD(v16[0]) - v8;
      v15 += LODWORD(v16[0]) - v8;
      if ( v2 < LODWORD(v16[0]) )
        v2 = v16[0];
      LOBYTE(v9) = LODWORD(v16[0]) == (_DWORD)v8;
      v3 &= v9;
      if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 4, v5, v6);
        v8 = *(unsigned int *)(a2 + 8);
        v7 = v16[0];
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v8) = v7;
      ++*(_DWORD *)(a2 + 8);
      if ( *(_DWORD *)(a1 + 64) != 4 )
      {
        if ( (unsigned __int8)sub_388AF10(a1, 9, "expected '}' here") )
          return 1;
        v10 = *(_DWORD *)(a2 + 8);
        if ( v10 <= 1 )
        {
          v18 = 1;
          v11 = "expected >= 2 uselistorder indexes";
        }
        else if ( v15 || v2 >= v10 )
        {
          v18 = 1;
          v11 = "expected distinct uselistorder indexes in range [0, size)";
        }
        else
        {
          if ( !(_BYTE)v3 )
            return v3;
          v18 = 1;
          v11 = "expected uselistorder indexes to change the order";
        }
        v16[0] = v11;
        v17 = 3;
        return (unsigned int)sub_38814C0(a1 + 8, v14, (__int64)v16);
      }
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    }
    return v4;
  }
}
