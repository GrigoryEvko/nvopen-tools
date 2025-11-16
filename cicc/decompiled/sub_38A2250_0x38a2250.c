// Function: sub_38A2250
// Address: 0x38a2250
//
__int64 __fastcall sub_38A2250(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v5; // r13d
  int v6; // eax
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rax
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = sub_388AF10(a1, 8, "expected '{' here");
  if ( (_BYTE)v5 )
    return v5;
  v6 = *(_DWORD *)(a1 + 64);
  if ( v6 == 9 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    return v5;
  }
  while ( 1 )
  {
    if ( v6 == 51 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      v13 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v13 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v11, v12);
        v13 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = 0;
      ++*(_DWORD *)(a2 + 8);
    }
    else
    {
      v5 = sub_38A2140(a1, v14, 0, a3, a4, a5);
      if ( (_BYTE)v5 )
        return v5;
      v9 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v7, v8);
        v9 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v9) = v14[0];
      ++*(_DWORD *)(a2 + 8);
    }
    if ( *(_DWORD *)(a1 + 64) != 4 )
      break;
    v6 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v6;
  }
  return sub_388AF10(a1, 9, "expected end of metadata node");
}
