// Function: sub_27272F0
// Address: 0x27272f0
//
__int64 __fastcall sub_27272F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // r13
  __int64 v8; // r15
  unsigned int v9; // r12d
  __int64 v10; // r15
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  *a1 = a4;
  a1[1] = a5;
  if ( !*(_BYTE *)(a3 + 192) )
    sub_CFDFC0(a3, a2, a3, a4, a5, a6);
  v7 = *(_QWORD *)(a3 + 16);
  v8 = 32LL * *(unsigned int *)(a3 + 24);
  v18 = v7 + v8;
  if ( v7 + v8 == v7 )
  {
    return 0;
  }
  else
  {
    v19 = *(_QWORD *)(a3 + 16);
    v9 = 0;
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v19 + 16);
        if ( v10 )
          break;
LABEL_5:
        v19 += 32;
        if ( v18 == v19 )
          return v9;
      }
      v11 = 0;
      while ( *(char *)(v10 + 7) < 0 )
      {
        v12 = sub_BD2BC0(v10);
        v14 = v12 + v13;
        v15 = 0;
        if ( *(char *)(v10 + 7) < 0 )
          v15 = sub_BD2BC0(v10);
        if ( v11 >= (unsigned int)((v14 - v15) >> 4) )
          goto LABEL_5;
        v16 = v11++;
        v9 |= sub_2726B80(a1, v10, v16);
      }
      v19 += 32;
    }
    while ( v18 != v19 );
  }
  return v9;
}
