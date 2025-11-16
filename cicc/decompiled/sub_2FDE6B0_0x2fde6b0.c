// Function: sub_2FDE6B0
// Address: 0x2fde6b0
//
__int64 __fastcall sub_2FDE6B0(int a1, _QWORD *a2, __int64 a3, int a4, __int64 a5, int a6)
{
  char v6; // r10
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r11
  int v11; // edx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  int v17; // ecx
  __int64 v19; // [rsp+0h] [rbp-8h]

  v6 = 0;
  v7 = a2[13];
  if ( v7 )
  {
    v8 = v7 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a3 + 16) + 6LL);
    v9 = (unsigned int)*(unsigned __int16 *)(v8 + 6) + a4;
    if ( *(unsigned __int16 *)(v8 + 8) > (unsigned int)v9 )
    {
      v10 = a2[11];
      v11 = *(_DWORD *)(v10 + 4 * v9);
      v12 = v7 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a5 + 16) + 6LL);
      v13 = (unsigned int)*(unsigned __int16 *)(v12 + 6) + a6;
      if ( *(unsigned __int16 *)(v12 + 8) > (unsigned int)v13 )
      {
        v14 = *(_DWORD *)(v10 + 4 * v13);
        if ( v11 + 1 >= v14 )
        {
          v15 = v11 - v14;
          a1 = v15 + 1;
          if ( v15 != -1 )
          {
            v16 = a2[12];
            v17 = *(_DWORD *)(v16 + 4 * v9);
            if ( v17 )
            {
              if ( v17 == *(_DWORD *)(v16 + 4 * v13) )
                a1 = v15;
            }
          }
          v6 = 1;
        }
      }
    }
  }
  LODWORD(v19) = a1;
  BYTE4(v19) = v6;
  return v19;
}
