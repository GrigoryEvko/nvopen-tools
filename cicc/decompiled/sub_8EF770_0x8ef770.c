// Function: sub_8EF770
// Address: 0x8ef770
//
void *__fastcall sub_8EF770(_DWORD *a1, _DWORD *a2, __int64 a3)
{
  __int64 v3; // r10
  int v5; // eax
  int v6; // edi
  int v7; // ebx
  int v8; // r14d
  int v9; // ebx
  bool v10; // sf
  int v11; // edi
  int v12; // r14d
  int v13; // r12d
  int v14; // r10d
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rax
  unsigned __int8 *p_dest; // r11
  int v18; // edi
  __int64 v19; // rcx
  unsigned int v20; // eax
  char *v21; // rbx
  unsigned int v22; // eax
  int v25; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v26; // [rsp+8h] [rbp-88h]
  int v27; // [rsp+10h] [rbp-80h]
  int v28; // [rsp+14h] [rbp-7Ch]
  __int64 dest; // [rsp+21h] [rbp-6Fh] BYREF
  int v31; // [rsp+29h] [rbp-67h]
  __int16 v32; // [rsp+2Dh] [rbp-63h]
  char v33; // [rsp+2Fh] [rbp-61h]
  _OWORD v34[2]; // [rsp+30h] [rbp-60h] BYREF
  char v35; // [rsp+50h] [rbp-40h]

  v3 = a3;
  v5 = *(_DWORD *)(a3 + 28);
  v6 = a2[7];
  v35 = 0;
  v7 = v5 + 14;
  v33 = 0;
  v8 = v6 + 14;
  v28 = v6;
  if ( v5 + 7 >= 0 )
    v7 = v5 + 7;
  dest = 0;
  v31 = 0;
  v9 = v7 >> 3;
  v10 = v6 + 7 < 0;
  v11 = v6 + 7;
  v34[0] = 0;
  if ( !v10 )
    v8 = v11;
  v34[1] = 0;
  v32 = 0;
  v12 = v8 >> 3;
  v27 = v9 + v12;
  if ( v5 > 0 )
  {
    memcpy((char *)v34 + v12, (const void *)(a3 + 12), (unsigned int)(v9 - 1) + 1LL);
    v3 = a3;
  }
  v13 = 8 * v27;
  sub_8EE740((char *)v34, 8 * v27, a2[2] - *(_DWORD *)(v3 + 8));
  v14 = a2[7];
  v15 = (unsigned __int8 *)(a2 + 3);
  if ( (v14 & 7) != 0 )
  {
    if ( v28 <= 0 )
    {
      p_dest = (unsigned __int8 *)&dest;
    }
    else
    {
      v25 = a2[7];
      v16 = (unsigned __int8 *)memcpy(&dest, v15, (unsigned int)(v12 - 1) + 1LL);
      v14 = v25;
      p_dest = v16;
    }
    v26 = p_dest;
    sub_8EE880(p_dest, 8 * v12, 8 - v14 % 8);
    sub_8EE880(v34, v13, 8 - a2[7] % 8);
    v15 = v26;
  }
  v18 = a2[2];
  if ( v28 <= 0 )
  {
    a1[2] = v18;
  }
  else
  {
    v19 = 0;
    v20 = 0;
    v21 = (char *)v34 + v9;
    do
    {
      v22 = v15[v19] + (unsigned __int8)v21[v19] + v20;
      v21[v19++] = v22;
      v20 = v22 >> 8;
    }
    while ( v12 > (int)v19 );
    a1[2] = v18;
    if ( v20 )
    {
      v13 += 8;
      a1[2] = v18 + 8;
      *((_BYTE *)v34 + v27) = 1;
    }
  }
  return sub_8EF4C0(a1, (char *)v34, v13);
}
