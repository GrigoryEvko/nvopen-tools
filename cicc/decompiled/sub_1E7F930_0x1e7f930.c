// Function: sub_1E7F930
// Address: 0x1e7f930
//
void __fastcall sub_1E7F930(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rdx
  unsigned __int64 v16; // [rsp-20h] [rbp-20h]

  if ( a2 != 1 )
  {
    v6 = 1;
    while ( *(_QWORD *)(a1 + 40LL * (unsigned int)(v6 + 1) + 24) != a4 )
    {
      v6 = (unsigned int)(v6 + 2);
      if ( (_DWORD)v6 == a2 )
        return;
    }
    v8 = 5 * v6;
    v9 = v6 << 32;
    v10 = *(unsigned int *)(a1 + 8 * v8 + 8);
    if ( (int)v10 < 0 )
      v11 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 16 * (v10 & 0x7FFFFFFF) + 8);
    else
      v11 = *(_QWORD *)(*(_QWORD *)(a5 + 272) + 8 * v10);
    if ( v11 )
    {
      if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
      {
        v11 = *(_QWORD *)(v11 + 32);
        if ( v11 )
        {
          if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
            BUG();
        }
      }
    }
    v12 = *(_QWORD *)(v11 + 16);
    v13 = (-858993459 * (unsigned int)((v11 - *(_QWORD *)(v12 + 32)) >> 3)) | (unsigned __int64)v9;
    v14 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v14 >= *(_DWORD *)(a3 + 12) )
    {
      v16 = v13;
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, a5, a6);
      v14 = *(unsigned int *)(a3 + 8);
      v13 = v16;
    }
    v15 = (_QWORD *)(*(_QWORD *)a3 + 16 * v14);
    *v15 = v12;
    v15[1] = v13;
    ++*(_DWORD *)(a3 + 8);
  }
}
