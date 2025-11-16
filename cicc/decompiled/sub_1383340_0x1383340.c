// Function: sub_1383340
// Address: 0x1383340
//
__int64 __fastcall sub_1383340(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // r10
  __int64 v10; // rax
  int v12; // edx
  int v13; // r11d

  v3 = *(unsigned int *)(a2 + 56);
  if ( (_DWORD)v3 )
  {
    v6 = *(_QWORD *)(a2 + 40);
    v7 = (v3 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a3 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v3) )
      {
        v10 = v8[1];
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v10;
        return a1;
      }
    }
    else
    {
      v12 = 1;
      while ( v9 != -8 )
      {
        v13 = v12 + 1;
        v7 = (v3 - 1) & (v12 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a3 == *v8 )
          goto LABEL_3;
        v12 = v13;
      }
    }
  }
  *(_BYTE *)(a1 + 8) = 0;
  return a1;
}
