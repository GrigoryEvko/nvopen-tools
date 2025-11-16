// Function: sub_326B110
// Address: 0x326b110
//
__int64 __fastcall sub_326B110(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  int v5; // eax
  int v6; // edx
  unsigned int *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h]
  __int64 v18; // [rsp+18h] [rbp-28h]

  v5 = *(_DWORD *)(a1 + 24);
  v15 = a4;
  v16 = a5;
  if ( v5 != 160 )
  {
    v6 = *(_DWORD *)(a2 + 24);
    if ( (v6 == 35 || v6 == 11) && v5 == 159 )
    {
      v8 = *(unsigned int **)(a1 + 40);
      v9 = *(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * v8[2];
      if ( *(_WORD *)v9 == (_WORD)v15 && (*(_QWORD *)(v9 + 8) == v16 || (_WORD)v15) )
      {
        v10 = *(_QWORD *)(a2 + 96);
        v11 = *(_QWORD *)(v10 + 24);
        if ( *(_DWORD *)(v10 + 32) > 0x40u )
          v11 = *(_QWORD *)v11;
        if ( (_WORD)v15 )
        {
          v12 = word_4456340[(unsigned __int16)v15 - 1];
          if ( !(v11 % v12) )
            return *(_QWORD *)&v8[10 * (unsigned int)(v11 / v12)];
        }
        else
        {
          v17 = sub_3007240((__int64)&v15);
          if ( !(v11 % (unsigned int)v17) )
          {
            v18 = sub_3007240((__int64)&v15);
            v12 = (unsigned int)v18;
            return *(_QWORD *)&v8[10 * (unsigned int)(v11 / v12)];
          }
        }
      }
    }
    return 0;
  }
  v13 = *(_QWORD *)(a1 + 40);
  v14 = *(_QWORD *)(*(_QWORD *)(v13 + 40) + 48LL) + 16LL * *(unsigned int *)(v13 + 48);
  if ( *(_WORD *)v14 != (_WORD)v15
    || *(_QWORD *)(v14 + 8) != v16 && !*(_WORD *)v14
    || *(_QWORD *)(v13 + 80) != a2
    || *(_DWORD *)(v13 + 88) != a3 )
  {
    return 0;
  }
  return *(_QWORD *)(v13 + 40);
}
