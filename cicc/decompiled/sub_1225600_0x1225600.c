// Function: sub_1225600
// Address: 0x1225600
//
__int64 __fastcall sub_1225600(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_120AFE0(a1, 8, "expected '{' here");
  if ( (_BYTE)v2 )
    return v2;
  v3 = *(_DWORD *)(a1 + 240);
  if ( v3 == 9 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    return v2;
  }
  while ( 1 )
  {
    if ( v3 == 54 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      v11 = *(unsigned int *)(a2 + 8);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 8u, v9, v10);
        v11 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = 0;
      ++*(_DWORD *)(a2 + 8);
    }
    else
    {
      v2 = sub_12254B0(a1, v12, 0);
      if ( (_BYTE)v2 )
        return v2;
      v6 = *(unsigned int *)(a2 + 8);
      v7 = v12[0];
      if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 8u, v4, v5);
        v6 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v6) = v7;
      ++*(_DWORD *)(a2 + 8);
    }
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    v3 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v3;
  }
  return sub_120AFE0(a1, 9, "expected end of metadata node");
}
