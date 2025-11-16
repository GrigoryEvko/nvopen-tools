// Function: sub_12A7DA0
// Address: 0x12a7da0
//
__int64 __fastcall sub_12A7DA0(__int64 a1, _QWORD *a2, unsigned __int64 *a3)
{
  char *v6; // r12
  __int64 v7; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 *v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-78h] BYREF
  char v17[16]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v18; // [rsp+20h] [rbp-60h]
  char v19[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v20; // [rsp+40h] [rbp-40h]

  v6 = sub_128F980((__int64)a2, *(_QWORD *)(a3[9] + 16));
  v7 = sub_127A030(a2[4] + 8LL, *a3, 0);
  v18 = 257;
  if ( v7 != *(_QWORD *)v6 )
  {
    if ( (unsigned __int8)v6[16] > 0x10u )
    {
      v20 = 257;
      v9 = sub_15FDBD0(47, v6, v7, v19, 0);
      v10 = a2[7];
      v6 = (char *)v9;
      if ( v10 )
      {
        v11 = (unsigned __int64 *)a2[8];
        sub_157E9D0(v10 + 40, v9);
        v12 = *((_QWORD *)v6 + 3);
        v13 = *v11;
        *((_QWORD *)v6 + 4) = v11;
        v13 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v6 + 3) = v13 | v12 & 7;
        *(_QWORD *)(v13 + 8) = v6 + 24;
        *v11 = *v11 & 7 | (unsigned __int64)(v6 + 24);
      }
      sub_164B780(v6, v17);
      v14 = a2[6];
      if ( v14 )
      {
        v16 = a2[6];
        sub_1623A60(&v16, v14, 2);
        if ( *((_QWORD *)v6 + 6) )
          sub_161E7C0(v6 + 48);
        v15 = v16;
        *((_QWORD *)v6 + 6) = v16;
        if ( v15 )
          sub_1623210(&v16, v15, v6 + 48);
      }
    }
    else
    {
      v6 = (char *)sub_15A46C0(47, v6, v7, 0);
    }
  }
  *(_QWORD *)a1 = v6;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
