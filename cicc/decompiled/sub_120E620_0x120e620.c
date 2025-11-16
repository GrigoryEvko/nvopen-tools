// Function: sub_120E620
// Address: 0x120e620
//
__int64 __fastcall sub_120E620(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r15
  int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  unsigned int v8; // r9d
  const char *v9; // rax
  unsigned __int64 v10; // rsi
  unsigned int v11; // [rsp+Ch] [rbp-64h]
  unsigned int v12[8]; // [rsp+10h] [rbp-60h] BYREF
  char v13; // [rsp+30h] [rbp-40h]
  char v14; // [rsp+31h] [rbp-3Fh]

  v3 = a1 + 176;
  *a3 = 0;
  if ( *(_DWORD *)(a1 + 240) != 4 )
  {
    v14 = 1;
    v9 = "expected ',' as start of index list";
LABEL_9:
    *(_QWORD *)v12 = v9;
    v10 = *(_QWORD *)(a1 + 232);
    v13 = 3;
    sub_11FD800(v3, v10, (__int64)v12, 1);
    return 1;
  }
  while ( 1 )
  {
    v5 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v5;
    if ( v5 == 511 )
      break;
    v12[0] = 0;
    result = sub_120BD00(a1, v12);
    if ( !(_BYTE)result )
    {
      v7 = *(unsigned int *)(a2 + 8);
      v8 = v12[0];
      if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v11 = v12[0];
        sub_C8D5F0(a2, (const void *)(a2 + 16), v7 + 1, 4u, v7 + 1, v12[0]);
        v7 = *(unsigned int *)(a2 + 8);
        result = 0;
        v8 = v11;
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v7) = v8;
      ++*(_DWORD *)(a2 + 8);
      if ( *(_DWORD *)(a1 + 240) == 4 )
        continue;
    }
    return result;
  }
  if ( !*(_DWORD *)(a2 + 8) )
  {
    v14 = 1;
    v9 = "expected index";
    goto LABEL_9;
  }
  *a3 = 1;
  return 0;
}
