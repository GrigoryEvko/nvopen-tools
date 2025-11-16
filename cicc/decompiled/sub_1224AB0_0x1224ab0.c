// Function: sub_1224AB0
// Address: 0x1224ab0
//
__int64 __fastcall sub_1224AB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rdx
  int v9; // eax
  _QWORD v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 240);
  if ( (unsigned int)v2 > 0xD )
  {
    if ( (_DWORD)v2 != 93 )
      goto LABEL_3;
    return 0;
  }
  v8 = 10880;
  if ( _bittest64(&v8, v2) )
    return 0;
LABEL_3:
  while ( 1 )
  {
    result = sub_1224A40((__int64 **)a1, v10);
    if ( (_BYTE)result )
      break;
    v6 = *(unsigned int *)(a2 + 8);
    v7 = v10[0];
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 8u, v4, v5);
      v6 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v6) = v7;
    ++*(_DWORD *)(a2 + 8);
    if ( *(_DWORD *)(a1 + 240) == 4 )
    {
      v9 = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = v9;
      if ( v9 != 93 )
        continue;
    }
    return 0;
  }
  return result;
}
