// Function: sub_3206C30
// Address: 0x3206c30
//
__int64 __fastcall sub_3206C30(__int64 a1, __int64 a2, int a3)
{
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  unsigned int v7; // eax
  unsigned __int8 v8; // bl
  unsigned __int8 v9; // bl
  __int16 v10; // ax
  int v11; // ecx
  unsigned __int64 v12; // rdx
  int v13; // eax
  bool v14; // zf
  __int64 v15; // rdi
  __int64 v16; // rax
  int v18; // edx
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  __int16 v20; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+12h] [rbp-3Eh]
  int v22; // [rsp+18h] [rbp-38h]
  char v23; // [rsp+22h] [rbp-2Eh]

  v5 = *(_BYTE *)(a2 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_QWORD *)(a2 - 32);
  else
    v6 = a2 - 16 - 8LL * ((v5 >> 2) & 0xF);
  v7 = sub_3206530(a1, *(unsigned __int8 **)(v6 + 24), 0);
  v19 = v7;
  v8 = v7;
  if ( v7 > 0xFFF || a3 | v7 & 0x700 || (unsigned __int16)sub_AF18C0(a2) != 15 )
  {
    v9 = 2 * (*(_QWORD *)(a2 + 24) == 64) + 10;
    v10 = sub_AF18C0(a2);
    if ( v10 == 16 )
    {
      v11 = 32;
    }
    else
    {
      v11 = 128;
      if ( v10 != 66 )
      {
        v11 = 0;
        if ( v10 != 15 )
          BUG();
      }
    }
    v12 = *(_QWORD *)(a2 + 24);
    v13 = a3;
    v14 = (*(_BYTE *)(a2 + 21) & 4) == 0;
    v15 = a1 + 648;
    v23 = 0;
    if ( !v14 )
    {
      BYTE1(v13) = BYTE1(a3) | 4;
      a3 = v13;
    }
    v20 = 4098;
    v21 = v19;
    v22 = a3 | v11 | v9 | ((unsigned __int8)(v12 >> 3) << 13);
    v16 = sub_3708FB0(v15, &v20);
    return sub_3707F80(a1 + 632, v16);
  }
  else
  {
    v18 = 1024;
    if ( *(_QWORD *)(a2 + 24) == 64 )
      v18 = 1536;
    return v18 | (unsigned int)v8;
  }
}
