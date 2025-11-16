// Function: sub_1E70E90
// Address: 0x1e70e90
//
void __fastcall sub_1E70E90(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  unsigned int v4; // ebx
  unsigned __int16 *v5; // r14
  __int64 v6; // r13
  unsigned __int16 *v7; // r12
  int v8; // esi
  signed int v9; // esi
  unsigned __int16 *v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdi
  unsigned int v13; // eax
  _DWORD *v15; // [rsp+8h] [rbp-38h]

  v4 = 0;
  v5 = (unsigned __int16 *)(a1[319] + ((unsigned __int64)*(unsigned int *)(a2 + 192) << 6));
  v6 = (__int64)(a1[384] - a1[383]) >> 2;
  v7 = v5 + 32;
  do
  {
    v8 = *v5;
    if ( !(_WORD)v8 )
      break;
    v9 = v8 - 1;
    if ( (_DWORD)v6 != v4 )
    {
      while ( 1 )
      {
        v10 = (unsigned __int16 *)(a1[383] + 4LL * v4);
        v11 = *v10 - 1;
        if ( v9 <= v11 )
          break;
        if ( (_DWORD)v6 == ++v4 )
          goto LABEL_8;
      }
      if ( v9 == v11 )
      {
        v13 = *(_DWORD *)(*a3 + 4LL * v9);
        if ( (int)v13 > (__int16)v10[1] && v13 <= 0x7FFF )
          v10[1] = v13;
      }
    }
LABEL_8:
    v12 = a1[284];
    if ( !*(_DWORD *)(*(_QWORD *)(v12 + 88) + 4LL * v9) )
    {
      v15 = (_DWORD *)(*(_QWORD *)(v12 + 88) + 4LL * v9);
      *v15 = sub_1ED7BB0();
    }
    v5 += 2;
  }
  while ( v7 != v5 );
}
