// Function: sub_AD7CA0
// Address: 0xad7ca0
//
__int64 __fastcall sub_AD7CA0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  _BYTE *v6; // r12
  unsigned int v7; // eax
  __int64 v8; // rsi
  unsigned int v9; // ebx
  __int64 v11; // rax
  _BYTE *v12; // rsi
  __int64 v13; // r12
  unsigned int v14; // ebx
  __int64 v15; // rdx
  int v16; // eax
  int v17; // ebx
  unsigned int v18; // r13d
  __int64 v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v21; // [rsp+8h] [rbp-28h]

  v6 = a1;
  while ( 1 )
  {
    if ( *v6 == 17 )
    {
      v7 = *((_DWORD *)v6 + 8);
      v8 = *((_QWORD *)v6 + 3);
      v9 = v7 - 1;
      if ( v7 <= 0x40 )
      {
        LOBYTE(v5) = v8 != 1LL << v9;
        return v5;
      }
      v5 = 1;
      if ( (*(_QWORD *)(v8 + 8LL * (v9 >> 6)) & (1LL << v9)) != 0 )
        LOBYTE(v5) = (unsigned int)sub_C44590(v6 + 24) != v9;
      return v5;
    }
    if ( *v6 == 18 )
    {
      v11 = sub_C33340(a1, a2, a3, a4, a5);
      v12 = v6 + 24;
      if ( *((_QWORD *)v6 + 3) == v11 )
        sub_C3E660(&v20, v12);
      else
        sub_C3A850(&v20, v12);
      v13 = v20;
      v14 = v21 - 1;
      if ( v21 <= 0x40 )
      {
        LOBYTE(v5) = v20 != 1LL << v14;
      }
      else
      {
        v5 = 1;
        if ( (*(_QWORD *)(v20 + 8LL * (v14 >> 6)) & (1LL << v14)) != 0 )
          LOBYTE(v5) = (unsigned int)sub_C44590(&v20) != v14;
        if ( v13 )
          j_j___libc_free_0_0(v13);
      }
      return v5;
    }
    v15 = *((_QWORD *)v6 + 1);
    v16 = *(unsigned __int8 *)(v15 + 8);
    if ( (_BYTE)v16 == 17 )
      break;
    if ( (unsigned int)(v16 - 17) <= 1 )
    {
      a1 = v6;
      a2 = 0;
      v6 = sub_AD7630((__int64)v6, 0, v15);
      if ( v6 )
        continue;
    }
    return 0;
  }
  v17 = *(_DWORD *)(v15 + 32);
  v18 = 0;
  if ( !v17 )
    return 1;
  while ( 1 )
  {
    v19 = sub_AD69F0(v6, v18);
    if ( !v19 || !(unsigned __int8)sub_AD7CA0(v19) )
      break;
    if ( ++v18 == v17 )
      return 1;
  }
  return 0;
}
