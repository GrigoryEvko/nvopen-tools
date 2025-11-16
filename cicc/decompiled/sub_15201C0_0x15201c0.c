// Function: sub_15201C0
// Address: 0x15201c0
//
void __fastcall sub_15201C0(__int64 a1, unsigned int a2, __int64 a3)
{
  _BYTE *v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rax
  int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int64 *v13; // r9
  unsigned __int64 v14; // r8
  unsigned int v15; // r11d
  unsigned int v16; // edx
  __int64 v17; // r10
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  char v21; // cl
  unsigned int v22[3]; // [rsp+Ch] [rbp-264h] BYREF
  __m128i v23; // [rsp+18h] [rbp-258h] BYREF
  unsigned __int64 v24; // [rsp+28h] [rbp-248h]
  __int64 *v25[2]; // [rsp+30h] [rbp-240h] BYREF
  _BYTE v26[560]; // [rsp+40h] [rbp-230h] BYREF

  v22[0] = a2;
  if ( a2 >= *(_DWORD *)(a1 + 8) )
    goto LABEL_5;
  v5 = *(_BYTE **)(*(_QWORD *)a1 + 8LL * a2);
  if ( !v5 )
    goto LABEL_5;
  if ( (unsigned __int8)(*v5 - 4) > 0x1Eu )
    BUG();
  if ( v5[1] == 2 )
  {
LABEL_5:
    v23.m128i_i64[1] = 0;
    v25[1] = (__int64 *)0x4000000000LL;
    v6 = *(_QWORD *)(a1 + 640) - *(_QWORD *)(a1 + 632);
    v25[0] = (__int64 *)v26;
    v24 = 0;
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 656) + 8 * (a2 - (v6 >> 4)));
    *(_DWORD *)(a1 + 320) = 0;
    v8 = (v7 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 304) = v8;
    v9 = v7 & 0x3F;
    if ( (v7 & 0x3F) == 0 )
    {
LABEL_6:
      v10 = sub_14ED070(a1 + 288, 0);
      v11 = sub_1510D70(a1 + 288, SHIDWORD(v10), (__int64)v25, (unsigned __int8 **)&v23.m128i_i64[1]);
      sub_151B070(&v23, (__int64 *)a1, v25, v11, a3, v22, v23.m128i_i64[1], v24);
      if ( (v23.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v23.m128i_i64[0] = v23.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_16BD130("Can't lazyload MD", 1);
      }
      if ( (_BYTE *)v25[0] != v26 )
        _libc_free((unsigned __int64)v25[0]);
      return;
    }
    v12 = *(_QWORD *)(a1 + 296);
    if ( v8 < v12 )
    {
      v13 = (unsigned __int64 *)(v8 + *(_QWORD *)(a1 + 288));
      if ( v12 >= v8 + 8 )
      {
        v14 = *v13;
        *(_QWORD *)(a1 + 304) = v8 + 8;
        v15 = 64;
LABEL_14:
        *(_DWORD *)(a1 + 320) = v15 - v9;
        *(_QWORD *)(a1 + 312) = v14 >> v9;
        goto LABEL_6;
      }
      *(_QWORD *)(a1 + 312) = 0;
      v16 = v12 - v8;
      v17 = v16;
      v15 = 8 * v16;
      v18 = v16 + v8;
      if ( v16 )
      {
        v19 = 0;
        v14 = 0;
        do
        {
          v20 = *((unsigned __int8 *)v13 + v19);
          v21 = 8 * v19++;
          v14 |= v20 << v21;
          *(_QWORD *)(a1 + 312) = v14;
        }
        while ( v17 != v19 );
        *(_QWORD *)(a1 + 304) = v18;
        *(_DWORD *)(a1 + 320) = v15;
        if ( v9 <= v15 )
          goto LABEL_14;
      }
      else
      {
        *(_QWORD *)(a1 + 304) = v18;
      }
    }
    sub_16BD130("Unexpected end of file", 1);
  }
}
