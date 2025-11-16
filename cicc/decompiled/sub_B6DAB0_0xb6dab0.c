// Function: sub_B6DAB0
// Address: 0xb6dab0
//
__int64 __fastcall sub_B6DAB0(int a1, __int64 a2)
{
  __int16 v3; // r14
  __int64 v4; // rbx
  _BYTE *v5; // rax
  char v6; // r15
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  _BYTE *v9; // r15
  _BYTE *v10; // rsi
  __int64 result; // rax
  int v12; // [rsp+1Ch] [rbp-74h] BYREF
  _BYTE *v13; // [rsp+20h] [rbp-70h] BYREF
  __int64 v14; // [rsp+28h] [rbp-68h]
  unsigned __int64 v15; // [rsp+30h] [rbp-60h]
  _BYTE v16[88]; // [rsp+38h] [rbp-58h] BYREF

  v3 = word_3F4CD80[a1 - 1];
  v13 = v16;
  v14 = 0;
  v15 = 40;
  if ( v3 < 0 )
  {
    v4 = 24808;
    v9 = &unk_3F46C80;
    v12 = v3 & 0x7FFF;
  }
  else
  {
    v12 = 0;
    v4 = 0;
    v5 = v16;
    v6 = v3 & 0xF;
    while ( 1 )
    {
      v5[v4] = v6;
      v8 = v14;
      v3 = (unsigned __int16)v3 >> 4;
      v4 = ++v14;
      if ( !v3 )
        break;
      v7 = v8 + 2;
      v6 = v3 & 0xF;
      if ( v7 > v15 )
      {
        sub_C8D290(&v13, v16, v7, 1);
        v4 = v14;
      }
      v5 = v13;
    }
    v12 = 0;
    v9 = v13;
  }
  v10 = v9;
  sub_B6B200((unsigned int *)&v12, (__int64)v9, v4, 0, a2);
  for ( result = (unsigned int)v12; v12 != v4; result = (unsigned int)v12 )
  {
    if ( !v9[result] )
      break;
    v10 = v9;
    sub_B6B200((unsigned int *)&v12, (__int64)v9, v4, 0, a2);
  }
  if ( v13 != v16 )
    return _libc_free(v13, v10);
  return result;
}
