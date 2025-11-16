// Function: sub_2EC6C00
// Address: 0x2ec6c00
//
void __fastcall sub_2EC6C00(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  unsigned int v3; // ebx
  unsigned __int16 *v4; // r14
  __int64 v5; // r13
  unsigned __int16 *v6; // r12
  int v7; // esi
  signed int v8; // esi
  unsigned __int16 *v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v13; // [rsp+18h] [rbp-38h]

  v3 = 0;
  v4 = (unsigned __int16 *)(a1[500] + ((unsigned __int64)*(unsigned int *)(a2 + 200) << 6));
  v5 = (__int64)(a1[613] - a1[612]) >> 2;
  v6 = v4 + 32;
  do
  {
    v7 = *v4;
    if ( !(_WORD)v7 )
      break;
    v8 = v7 - 1;
    if ( (_DWORD)v5 != v3 )
    {
      while ( 1 )
      {
        v9 = (unsigned __int16 *)(a1[612] + 4LL * v3);
        v10 = *v9 - 1;
        if ( v8 <= v10 )
          break;
        if ( (_DWORD)v5 == ++v3 )
          goto LABEL_8;
      }
      if ( v8 == v10 )
      {
        v11 = *(_DWORD *)(*a3 + 4LL * v8);
        if ( (int)v11 > (__int16)v9[1] && v11 <= 0x7FFF )
          v9[1] = v11;
      }
    }
LABEL_8:
    if ( !*(_DWORD *)(*(_QWORD *)(a1[443] + 296LL) + 4LL * v8) )
    {
      v13 = a1[443];
      *(_DWORD *)(*(_QWORD *)(v13 + 296) + 4LL * v8) = sub_2F60A40(v13);
    }
    v4 += 2;
  }
  while ( v6 != v4 );
}
