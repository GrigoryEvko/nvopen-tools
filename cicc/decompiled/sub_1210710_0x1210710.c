// Function: sub_1210710
// Address: 0x1210710
//
char __fastcall sub_1210710(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  int v3; // r12d
  char result; // al
  __int64 v5; // r8
  __int64 v6; // r9
  int v7; // r10d
  __int64 v8; // rax
  unsigned int v9; // edx
  bool v10; // dl
  const char *v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // [rsp+10h] [rbp-70h]
  int v14; // [rsp+18h] [rbp-68h]
  char v15; // [rsp+1Fh] [rbp-61h]
  _QWORD v16[4]; // [rsp+20h] [rbp-60h] BYREF
  char v17; // [rsp+40h] [rbp-40h]
  char v18; // [rsp+41h] [rbp-3Fh]

  v13 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_120AFE0(a1, 8, "expected '{' here") )
    return 1;
  if ( *(_DWORD *)(a1 + 240) == 9 )
  {
    v12 = *(_QWORD *)(a1 + 232);
    v18 = 1;
    v16[0] = "expected non-empty list of uselistorder indexes";
    v17 = 3;
    sub_11FD800(a1 + 176, v12, (__int64)v16, 1);
    return 1;
  }
  v15 = 1;
  v2 = 0;
  v3 = 0;
  while ( 1 )
  {
    result = sub_120BD00(a1, v16);
    if ( result )
      return result;
    v7 = v16[0];
    v8 = *(unsigned int *)(a2 + 8);
    v3 += LODWORD(v16[0]) - v8;
    if ( v2 < LODWORD(v16[0]) )
      v2 = v16[0];
    v15 &= LODWORD(v16[0]) == (_DWORD)v8;
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v14 = v16[0];
      sub_C8D5F0(a2, (const void *)(a2 + 16), v8 + 1, 4u, v5, v6);
      v8 = *(unsigned int *)(a2 + 8);
      v7 = v14;
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v8) = v7;
    ++*(_DWORD *)(a2 + 8);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  if ( (unsigned __int8)sub_120AFE0(a1, 9, "expected '}' here") )
    return 1;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 <= 1 )
  {
    v18 = 1;
    v17 = 3;
    v16[0] = "expected >= 2 uselistorder indexes";
    sub_11FD800(a1 + 176, v13, (__int64)v16, 1);
    return 1;
  }
  v10 = v2 >= v9;
  result = v10 || v3 != 0;
  if ( result )
  {
    v18 = 1;
    v15 = v10 || v3 != 0;
    v11 = "expected distinct uselistorder indexes in range [0, size)";
LABEL_14:
    v16[0] = v11;
    v17 = 3;
    sub_11FD800(a1 + 176, v13, (__int64)v16, 1);
    return v15;
  }
  if ( v15 )
  {
    v18 = 1;
    v11 = "expected uselistorder indexes to change the order";
    goto LABEL_14;
  }
  return result;
}
