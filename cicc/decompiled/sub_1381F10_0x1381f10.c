// Function: sub_1381F10
// Address: 0x1381f10
//
__int64 __fastcall sub_1381F10(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rax
  unsigned int v6; // ecx
  unsigned int v9; // r9d
  __int64 v10; // rdi
  __int64 *v11; // rdx
  __int64 v12; // r11
  __int64 v13; // rdi
  int v15; // edx
  int v16; // ebx

  v4 = *(unsigned int *)(a2 + 24);
  v6 = a4 + 1;
  if ( (_DWORD)v4 )
  {
    v9 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = *(_QWORD *)(a2 + 8);
    v11 = (__int64 *)(v10 + 32LL * v9);
    v12 = *v11;
    if ( a3 == *v11 )
    {
LABEL_3:
      if ( v11 != (__int64 *)(v10 + 32 * v4) )
      {
        v13 = v11[1];
        if ( -1227133513 * (unsigned int)((v11[2] - v13) >> 3) > v6 )
        {
          if ( v13 + 56LL * v6 )
          {
            *(_BYTE *)(a1 + 16) = 1;
            *(_QWORD *)a1 = a3;
            *(_DWORD *)(a1 + 8) = v6;
            return a1;
          }
        }
      }
    }
    else
    {
      v15 = 1;
      while ( v12 != -8 )
      {
        v16 = v15 + 1;
        v9 = (v4 - 1) & (v15 + v9);
        v11 = (__int64 *)(v10 + 32LL * v9);
        v12 = *v11;
        if ( a3 == *v11 )
          goto LABEL_3;
        v15 = v16;
      }
    }
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
}
