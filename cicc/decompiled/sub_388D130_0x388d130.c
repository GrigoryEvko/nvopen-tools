// Function: sub_388D130
// Address: 0x388d130
//
__int64 __fastcall sub_388D130(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r15
  int v5; // eax
  __int64 result; // rax
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rdx
  const char *v10; // rax
  unsigned __int64 v11; // rsi
  _QWORD v12[2]; // [rsp+10h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-40h]
  char v14; // [rsp+21h] [rbp-3Fh]

  v3 = a1 + 8;
  *a3 = 0;
  if ( *(_DWORD *)(a1 + 64) != 4 )
  {
    v14 = 1;
    v10 = "expected ',' as start of index list";
LABEL_9:
    v11 = *(_QWORD *)(a1 + 56);
    v12[0] = v10;
    v13 = 3;
    return sub_38814C0(v3, v11, (__int64)v12);
  }
  while ( 1 )
  {
    v5 = sub_3887100(v3);
    *(_DWORD *)(a1 + 64) = v5;
    if ( v5 == 376 )
      break;
    LODWORD(v12[0]) = 0;
    result = sub_388BA90(a1, v12);
    if ( !(_BYTE)result )
    {
      v9 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 4, v7, v8);
        v9 = *(unsigned int *)(a2 + 8);
        result = 0;
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v9) = v12[0];
      ++*(_DWORD *)(a2 + 8);
      if ( *(_DWORD *)(a1 + 64) == 4 )
        continue;
    }
    return result;
  }
  if ( !*(_DWORD *)(a2 + 8) )
  {
    v14 = 1;
    v10 = "expected index";
    goto LABEL_9;
  }
  *a3 = 1;
  return 0;
}
