// Function: sub_255E1F0
// Address: 0x255e1f0
//
__int64 __fastcall sub_255E1F0(__int64 a1, unsigned __int64 a2, char a3)
{
  __int64 v4; // r10
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rsi
  int v8; // r11d
  unsigned int i; // eax
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE v16[8]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v17; // [rsp+8h] [rbp-18h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)a1;
  if ( v4 )
  {
    if ( !v5 )
    {
      if ( !*(_BYTE *)(a1 + 16) && !a3 )
      {
        sub_B8BA60((__int64)v16, *(_QWORD *)(v4 + 8), v4, (__int64)&unk_4F92384, a2);
        return (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v17 + 104LL))(v17, &unk_4F92384) + 184;
      }
      v13 = sub_B82360(*(_QWORD *)(v4 + 8), (__int64)&unk_4F92384);
      if ( v13 )
      {
        v14 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v13 + 104LL))(v13, &unk_4F92384);
        if ( v14 )
          return v14 + 184;
      }
      return 0;
    }
  }
  else if ( !v5 )
  {
    return 0;
  }
  if ( !*(_BYTE *)(a1 + 16) && !a3 )
  {
    v15 = sub_BC1CD0(*(_QWORD *)a1, &unk_4F92388, a2);
    return v15 + 8;
  }
  v6 = *(unsigned int *)(v5 + 88);
  v7 = *(_QWORD *)(v5 + 72);
  if ( !(_DWORD)v6 )
    return 0;
  v8 = 1;
  for ( i = (v6 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F92388 >> 9) ^ ((unsigned int)&unk_4F92388 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v6 - 1) & v11 )
  {
    v10 = v7 + 24LL * i;
    if ( *(_UNKNOWN **)v10 == &unk_4F92388 && a2 == *(_QWORD *)(v10 + 8) )
      break;
    if ( *(_QWORD *)v10 == -4096 && *(_QWORD *)(v10 + 8) == -4096 )
      return 0;
    v11 = v8 + i;
    ++v8;
  }
  if ( v10 == v7 + 24 * v6 )
    return 0;
  v15 = *(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL);
  if ( v15 )
    return v15 + 8;
  return 0;
}
