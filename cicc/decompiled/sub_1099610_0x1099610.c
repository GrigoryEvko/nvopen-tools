// Function: sub_1099610
// Address: 0x1099610
//
__int64 __fastcall sub_1099610(__int64 a1, __int64 a2, char *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r11
  unsigned __int8 *v8; // rdi
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int8 *v11; // r11
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r13
  _QWORD v16[4]; // [rsp+0h] [rbp-A0h] BYREF
  __int16 v17; // [rsp+20h] [rbp-80h]
  char *v18; // [rsp+30h] [rbp-70h] BYREF
  __int64 v19; // [rsp+38h] [rbp-68h]
  _OWORD v20[3]; // [rsp+40h] [rbp-60h] BYREF
  int v21; // [rsp+70h] [rbp-30h]

  v6 = (unsigned __int64)a3;
  v8 = (unsigned __int8 *)a2;
  v18 = (char *)v20;
  v19 = 0x600000004LL;
  v21 = 256;
  memset(v20, 0, 32);
  if ( (unsigned __int64)a3 <= 2 )
  {
LABEL_10:
    v11 = &v8[v6];
    if ( v11 != v8 )
    {
      a2 = 1;
      do
      {
        a4 = *v8++;
        a3 = (char *)(1LL << a4);
        *(_QWORD *)&v18[(a4 >> 3) & 0x18] |= 1LL << a4;
      }
      while ( v8 != v11 );
    }
    *(_BYTE *)(a1 + 72) = *(_BYTE *)(a1 + 72) & 0xFC | 2;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x600000000LL;
    if ( (_DWORD)v19 )
    {
      a2 = (__int64)&v18;
      sub_10992B0(a1, &v18, (__int64)a3, a4, a5, a6);
    }
    *(_DWORD *)(a1 + 64) = v21;
  }
  else
  {
    v9 = a4;
    while ( 1 )
    {
      while ( 1 )
      {
        a4 = *v8;
        if ( v8[1] == 45 )
          break;
        --v6;
        ++v8;
        a3 = (char *)(1LL << a4);
        *(_QWORD *)&v18[(a4 >> 3) & 0x18] |= 1LL << a4;
        if ( v6 <= 2 )
          goto LABEL_10;
      }
      a6 = v8[2];
      if ( (unsigned __int8)a4 > (unsigned __int8)a6 )
        break;
      if ( (int)a4 <= (int)a6 )
      {
        a6 = (unsigned int)(a6 + 1);
        do
        {
          a3 = v18;
          a2 = 1LL << a4;
          v10 = (unsigned int)a4 >> 6;
          a4 = (unsigned int)(a4 + 1);
          *(_QWORD *)&v18[8 * v10] |= a2;
        }
        while ( (_DWORD)a4 != (_DWORD)a6 );
      }
      v6 -= 3LL;
      v8 += 3;
      if ( v6 <= 2 )
        goto LABEL_10;
    }
    v16[2] = v9;
    v17 = 1283;
    v16[3] = a5;
    v16[0] = "invalid glob pattern: ";
    v13 = sub_2241E50(v8, a2, 1283, a4, a5);
    v14 = sub_22077B0(64);
    v15 = v14;
    if ( v14 )
    {
      a2 = (__int64)v16;
      sub_C63EB0(v14, (__int64)v16, 22, v13);
    }
    *(_BYTE *)(a1 + 72) |= 3u;
    *(_QWORD *)a1 = v15 & 0xFFFFFFFFFFFFFFFELL;
  }
  if ( v18 != (char *)v20 )
    _libc_free(v18, a2);
  return a1;
}
