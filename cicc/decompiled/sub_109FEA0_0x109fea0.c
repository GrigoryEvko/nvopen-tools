// Function: sub_109FEA0
// Address: 0x109fea0
//
unsigned __int8 *__fastcall sub_109FEA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const char **a4,
        __int64 a5,
        unsigned __int16 a6,
        __int64 a7)
{
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax

  v9 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v10 = v9;
  if ( v9 )
  {
    sub_B44260((__int64)v9, *(_QWORD *)(a2 + 8), 57, 3u, a5, a6);
    if ( *((_QWORD *)v10 - 12) )
    {
      v11 = *((_QWORD *)v10 - 11);
      **((_QWORD **)v10 - 10) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *((_QWORD *)v10 - 10);
    }
    *((_QWORD *)v10 - 12) = a1;
    if ( a1 )
    {
      v12 = *(_QWORD *)(a1 + 16);
      *((_QWORD *)v10 - 11) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = v10 - 88;
      *((_QWORD *)v10 - 10) = a1 + 16;
      *(_QWORD *)(a1 + 16) = v10 - 96;
    }
    if ( *((_QWORD *)v10 - 8) )
    {
      v13 = *((_QWORD *)v10 - 7);
      **((_QWORD **)v10 - 6) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *((_QWORD *)v10 - 6);
    }
    *((_QWORD *)v10 - 8) = a2;
    v14 = *(_QWORD *)(a2 + 16);
    *((_QWORD *)v10 - 7) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = v10 - 56;
    *((_QWORD *)v10 - 6) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v10 - 64;
    if ( *((_QWORD *)v10 - 4) )
    {
      v15 = *((_QWORD *)v10 - 3);
      **((_QWORD **)v10 - 2) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *((_QWORD *)v10 - 2);
    }
    *((_QWORD *)v10 - 4) = a3;
    if ( a3 )
    {
      v16 = *(_QWORD *)(a3 + 16);
      *((_QWORD *)v10 - 3) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = v10 - 24;
      *((_QWORD *)v10 - 2) = a3 + 16;
      *(_QWORD *)(a3 + 16) = v10 - 32;
    }
    sub_BD6B50(v10, a4);
  }
  if ( a7 )
    sub_B47C00((__int64)v10, a7, 0, 0);
  return v10;
}
