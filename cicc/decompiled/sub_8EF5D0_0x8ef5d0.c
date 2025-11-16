// Function: sub_8EF5D0
// Address: 0x8ef5d0
//
unsigned __int64 __fastcall sub_8EF5D0(_DWORD *a1, _DWORD *a2)
{
  bool v2; // zf
  unsigned __int64 result; // rax
  int v4; // ecx
  int v5; // r15d
  int v6; // r14d
  int v9; // r15d
  int v10; // r14d
  int v11; // esi
  int v12; // esi
  int v13; // eax
  int v14; // r14d
  char *v15; // rcx
  _BYTE *v16; // rsi
  int v17; // edi
  unsigned int v18; // edx
  char v19; // al
  int v20; // [rsp+Ch] [rbp-94h]
  int v21; // [rsp+Ch] [rbp-94h]
  _OWORD v22[2]; // [rsp+10h] [rbp-90h] BYREF
  char v23; // [rsp+30h] [rbp-70h]
  _OWORD v24[2]; // [rsp+40h] [rbp-60h] BYREF
  char v25; // [rsp+60h] [rbp-40h]
  char v26; // [rsp+61h] [rbp-3Fh] BYREF

  v2 = *a2 == 6;
  result = (unsigned int)a2[7];
  v23 = 0;
  v25 = 0;
  v4 = a1[7];
  memset(v22, 0, sizeof(v22));
  memset(v24, 0, sizeof(v24));
  if ( !v2 )
  {
    v5 = result + 14;
    v6 = v4 + 14;
    if ( (int)result + 7 >= 0 )
      v5 = result + 7;
    v9 = v5 >> 3;
    if ( v4 + 7 >= 0 )
      v6 = v4 + 7;
    v10 = v6 >> 3;
    if ( (int)result <= 0 )
    {
      v12 = v10;
    }
    else
    {
      v20 = v4;
      memcpy((char *)v24 + v10, a2 + 3, (unsigned int)(v9 - 1) + 1LL);
      v11 = 1;
      v4 = v20;
      if ( v9 > 0 )
        v11 = v9;
      v12 = v10 + v11;
    }
    v21 = v4;
    sub_8EE740((char *)v24, 8 * v12, a1[2] - a2[2]);
    if ( v21 > 0 )
      memcpy((char *)v22 + v9, a1 + 3, (unsigned int)(v10 - 1) + 1LL);
    v13 = a1[7];
    v14 = 8 * (v9 + v10);
    if ( (v13 & 7) != 0 )
    {
      sub_8EE880(v22, v14, 8 - v13 % 8);
      sub_8EE880(v24, v14, 8 - a1[7] % 8);
    }
    v15 = (char *)v24;
    v16 = v22;
    v17 = 0;
    do
    {
      v18 = v17 + (unsigned __int8)*v15;
      LOBYTE(v17) = (unsigned __int8)*v16 < v18;
      v19 = *v16 - v18;
      ++v15;
      *v16++ = v19;
      v17 = (unsigned __int8)v17;
    }
    while ( &v26 != v15 );
    return (unsigned __int64)sub_8EF4C0(a1, (char *)v22, v14);
  }
  return result;
}
