// Function: sub_AD7B80
// Address: 0xad7b80
//
bool __fastcall sub_AD7B80(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // r12
  bool result; // al
  __int64 v7; // rax
  _BYTE *v8; // rsi
  __int64 v9; // r13
  unsigned int v10; // ebx
  __int64 v11; // rdx
  int v12; // eax
  int v13; // r13d
  unsigned int v14; // ebx
  __int64 v15; // rdi
  bool v16; // [rsp+Fh] [rbp-31h]
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-28h]

  v5 = a1;
  while ( 1 )
  {
    if ( *v5 == 17 )
      return !sub_AD7A80(v5, a2, a3, a4, a5);
    if ( *v5 == 18 )
      break;
    v11 = *((_QWORD *)v5 + 1);
    v12 = *(unsigned __int8 *)(v11 + 8);
    if ( (_BYTE)v12 == 17 )
    {
      v13 = *(_DWORD *)(v11 + 32);
      v14 = 0;
      if ( !v13 )
        return 1;
      while ( 1 )
      {
        v15 = sub_AD69F0(v5, v14);
        if ( !v15 || !(unsigned __int8)sub_AD7B80(v15) )
          break;
        if ( v13 == ++v14 )
          return 1;
      }
      return 0;
    }
    if ( (unsigned int)(v12 - 17) <= 1 )
    {
      a1 = v5;
      a2 = 0;
      v5 = sub_AD7630((__int64)v5, 0, v11);
      if ( v5 )
        continue;
    }
    return 0;
  }
  v7 = sub_C33340(a1, a2, a3, a4, a5);
  v8 = v5 + 24;
  if ( *((_QWORD *)v5 + 3) == v7 )
    sub_C3E660(&v17, v8);
  else
    sub_C3A850(&v17, v8);
  v9 = v17;
  v10 = v18;
  result = v17 != 1;
  if ( v18 > 0x40 )
  {
    result = v10 - 1 != (unsigned int)sub_C444A0(&v17);
    if ( v9 )
    {
      v16 = result;
      j_j___libc_free_0_0(v9);
      return v16;
    }
  }
  return result;
}
