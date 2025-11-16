// Function: sub_2DDC390
// Address: 0x2ddc390
//
unsigned __int64 __fastcall sub_2DDC390(unsigned int **a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r11
  __int64 v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 v14; // r11
  unsigned int v15; // eax
  __int64 v16; // r9
  __int64 v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-88h]
  unsigned int v21; // [rsp+8h] [rbp-88h]
  unsigned int v22; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  __int64 v24; // [rsp+18h] [rbp-78h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  unsigned int v26; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v27[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v28; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_BYTE *)(v4 + 8);
  if ( v5 == 15 )
  {
    v26 = 0;
    v7 = sub_ACADE0((__int64 **)a3);
    v21 = *(_DWORD *)(v4 + 12);
    if ( v21 )
    {
      v14 = a2;
      v15 = 0;
      do
      {
        v25 = v14;
        v16 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8LL * v15);
        v28 = 257;
        v23 = v16;
        v17 = sub_94D3D0(a1, v14, (__int64)&v26, 1, (__int64)v27);
        v18 = (_BYTE *)sub_2DDC390(a1, v17, v23);
        v28 = 257;
        v19 = sub_2466140((__int64 *)a1, v7, v18, &v26, 1, (__int64)v27);
        v14 = v25;
        v7 = v19;
        v15 = ++v26;
      }
      while ( v26 < v21 );
    }
  }
  else if ( v5 == 16 )
  {
    v6 = 0;
    if ( *(_BYTE *)(a3 + 8) == 16 )
      v6 = a3;
    v20 = v6;
    v26 = 0;
    v7 = sub_ACADE0((__int64 **)a3);
    v22 = *(_QWORD *)(v4 + 32);
    if ( v22 )
    {
      v8 = a2;
      do
      {
        v24 = v8;
        v9 = *(_QWORD *)(v20 + 24);
        v28 = 257;
        v10 = sub_94D3D0(a1, v8, (__int64)&v26, 1, (__int64)v27);
        v11 = (_BYTE *)sub_2DDC390(a1, v10, v9);
        v28 = 257;
        v12 = sub_2466140((__int64 *)a1, v7, v11, &v26, 1, (__int64)v27);
        v8 = v24;
        v7 = v12;
        ++v26;
      }
      while ( v26 < v22 );
    }
  }
  else
  {
    if ( v5 == 12 )
    {
      if ( *(_BYTE *)(a3 + 8) == 14 )
      {
        v28 = 257;
        return sub_2DDC1F0((__int64 *)a1, 0x30u, a2, (__int64 **)a3, (__int64)v27, 0, v26, 0);
      }
    }
    else if ( v5 == 14 && *(_BYTE *)(a3 + 8) == 12 )
    {
      v28 = 257;
      return sub_2DDC1F0((__int64 *)a1, 0x2Fu, a2, (__int64 **)a3, (__int64)v27, 0, v26, 0);
    }
    v28 = 257;
    return sub_2DDC1F0((__int64 *)a1, 0x31u, a2, (__int64 **)a3, (__int64)v27, 0, v26, 0);
  }
  return v7;
}
