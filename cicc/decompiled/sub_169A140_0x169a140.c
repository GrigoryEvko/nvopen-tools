// Function: sub_169A140
// Address: 0x169a140
//
__int64 __fastcall sub_169A140(__int16 **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v5; // r13d
  __int64 v6; // r15
  unsigned int v7; // r9d
  unsigned int v8; // ecx
  __int16 v9; // ax
  unsigned int v10; // r13d
  unsigned int v11; // eax
  int v12; // r10d
  __int64 v13; // rcx
  unsigned int v14; // r9d
  unsigned int v15; // r10d
  int v17; // eax
  unsigned int v18; // [rsp+4h] [rbp-3Ch]
  unsigned int v19; // [rsp+8h] [rbp-38h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  v20 = a3;
  *((_BYTE *)a1 + 18) = *((_BYTE *)a1 + 18) & 0xF8 | 2;
  v5 = sub_16A7150(a2, (unsigned int)a3, a3) + 1;
  v6 = sub_1698470((__int64)a1);
  v7 = sub_1698310((__int64)a1);
  v8 = *((_DWORD *)*a1 + 1);
  if ( v5 < v8 )
  {
    *((_WORD *)a1 + 8) = v8 - 1;
    sub_16A8750(v6, v7, a2, v5, 0);
    v15 = 0;
  }
  else
  {
    v9 = v5 - 1;
    v10 = v5 - v8;
    *((_WORD *)a1 + 8) = v9;
    v18 = v7;
    v19 = v8;
    v11 = sub_16A7110(a2, v20);
    v12 = 0;
    v13 = v19;
    v14 = v18;
    if ( v10 > v11 )
    {
      if ( v10 == v11 + 1 )
      {
        v12 = 2;
      }
      else if ( v10 > v20 << 6 || (v17 = sub_16A70B0(a2, v10 - 1), v14 = v18, v13 = v19, v12 = 3, !v17) )
      {
        v12 = 1;
      }
    }
    v21 = v12;
    sub_16A8750(v6, v14, a2, v13, v10);
    v15 = v21;
  }
  return sub_1698EC0(a1, a4, v15);
}
