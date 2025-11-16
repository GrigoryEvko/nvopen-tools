// Function: sub_1F4E120
// Address: 0x1f4e120
//
__int64 __fastcall sub_1F4E120(char *a1, int a2, __int64 a3)
{
  char *v5; // rax
  _BYTE *v6; // rsi
  __int64 v7; // r8
  __int64 result; // rax
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // cl
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v5 = sub_1DCC790(a1, a2);
  v13[0] = a3;
  v6 = sub_1F4C640(*((_QWORD **)v5 + 4), *((_QWORD *)v5 + 5), v13);
  result = 0;
  if ( *(_BYTE **)(v7 + 40) != v6 )
  {
    sub_1DCBB50(v7 + 32, v6);
    v9 = *(_DWORD *)(a3 + 40);
    v10 = *(_QWORD *)(a3 + 32);
    if ( v9 )
    {
      v11 = v10 + 40LL * (unsigned int)(v9 - 1) + 40;
      while ( 1 )
      {
        if ( !*(_BYTE *)v10 )
        {
          v12 = *(_BYTE *)(v10 + 3);
          if ( (v12 & 0x10) != 0 && a2 == *(_DWORD *)(v10 + 8) )
            break;
        }
        v10 += 40;
        if ( v10 == v11 )
          return 1;
      }
      *(_BYTE *)(v10 + 3) = v12 & 0xBF;
      return 1;
    }
    else
    {
      return 1;
    }
  }
  return result;
}
