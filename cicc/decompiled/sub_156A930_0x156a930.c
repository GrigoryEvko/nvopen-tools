// Function: sub_156A930
// Address: 0x156a930
//
__int64 __fastcall sub_156A930(__int64 *a1, _BYTE *a2, unsigned int a3)
{
  _QWORD *v4; // r12
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 *v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-78h] BYREF
  _DWORD v19[4]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v20; // [rsp+20h] [rbp-60h]
  _QWORD v21[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  v4 = a2;
  v6 = *(_DWORD *)(*(_QWORD *)a2 + 8LL);
  v7 = sub_1643320(a1[3]);
  v8 = sub_16463B0(v7, v6 >> 8);
  v20 = 257;
  if ( v8 != *(_QWORD *)a2 )
  {
    if ( a2[16] > 0x10u )
    {
      v22 = 257;
      v11 = sub_15FDBD0(47, a2, v8, v21, 0);
      v12 = a1[1];
      v4 = (_QWORD *)v11;
      if ( v12 )
      {
        v13 = (unsigned __int64 *)a1[2];
        sub_157E9D0(v12 + 40, v11);
        v14 = v4[3];
        v15 = *v13;
        v4[4] = v13;
        v15 &= 0xFFFFFFFFFFFFFFF8LL;
        v4[3] = v15 | v14 & 7;
        *(_QWORD *)(v15 + 8) = v4 + 3;
        *v13 = *v13 & 7 | (unsigned __int64)(v4 + 3);
      }
      sub_164B780(v4, v19);
      v16 = *a1;
      if ( *a1 )
      {
        v18 = *a1;
        sub_1623A60(&v18, v16, 2);
        if ( v4[6] )
          sub_161E7C0(v4 + 6);
        v17 = v18;
        v4[6] = v18;
        if ( v17 )
          sub_1623210(&v18, v17, v4 + 6);
      }
    }
    else
    {
      v4 = (_QWORD *)sub_15A46C0(47, a2, v8, 0);
    }
  }
  if ( a3 <= 7 )
  {
    if ( a3 )
    {
      v9 = 0;
      do
      {
        v19[v9] = v9;
        ++v9;
      }
      while ( a3 != (_DWORD)v9 );
    }
    v21[0] = "extract";
    v22 = 259;
    return sub_156A7D0(a1, (__int64)v4, (__int64)v4, (__int64)v19, a3, (__int64)v21);
  }
  return (__int64)v4;
}
