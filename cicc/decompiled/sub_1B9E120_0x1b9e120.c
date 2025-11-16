// Function: sub_1B9E120
// Address: 0x1b9e120
//
void __fastcall sub_1B9E120(__int64 a1, __int64 a2)
{
  unsigned int v4; // r13d
  unsigned int v5; // ebx
  __int64 v6; // rdi
  char v7; // cl
  unsigned __int64 v8; // rsi
  __int64 **v9; // rax
  __int64 v10; // rax
  unsigned int *v11; // r8
  int v12; // r9d
  unsigned int v13; // [rsp+Ch] [rbp-44h]
  unsigned int v14[14]; // [rsp+18h] [rbp-38h] BYREF

  if ( *(_BYTE *)(a2 + 16) )
  {
    sub_1B9DE60(*(_QWORD *)(a2 + 224), *(_QWORD *)(a1 + 40), (unsigned int *)(a2 + 8), *(_BYTE *)(a1 + 49));
    if ( *(_BYTE *)(a1 + 50) && *(_DWORD *)a2 > 1u )
    {
      if ( !*(_DWORD *)(a2 + 12) )
      {
        v9 = (__int64 **)sub_16463B0(**(__int64 ***)(a1 + 40), *(_DWORD *)a2);
        v10 = sub_1599EF0(v9);
        sub_1B99BD0(*(unsigned int **)(a2 + 184), *(_QWORD *)(a1 + 40), *(_DWORD *)(a2 + 8), v10, v11, v12);
      }
      sub_1B99F70(*(__int64 **)(a2 + 224), *(unsigned __int64 **)(a1 + 40), (unsigned int *)(a2 + 8));
    }
  }
  else
  {
    v13 = 1;
    if ( !*(_BYTE *)(a1 + 48) )
      v13 = *(_DWORD *)a2;
    if ( *(_DWORD *)(a2 + 4) )
    {
      v4 = 0;
      do
      {
        v5 = 0;
        if ( v13 )
        {
          do
          {
            v6 = *(_QWORD *)(a2 + 224);
            v7 = *(_BYTE *)(a1 + 49);
            v14[1] = v5;
            v8 = *(_QWORD *)(a1 + 40);
            v14[0] = v4;
            ++v5;
            sub_1B9DE60(v6, v8, v14, v7);
          }
          while ( v13 != v5 );
        }
        ++v4;
      }
      while ( *(_DWORD *)(a2 + 4) > v4 );
    }
  }
}
