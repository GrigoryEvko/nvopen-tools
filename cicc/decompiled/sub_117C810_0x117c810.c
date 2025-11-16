// Function: sub_117C810
// Address: 0x117c810
//
unsigned __int8 *__fastcall sub_117C810(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v5; // rsi
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int8 *v9; // r12
  unsigned __int16 v10; // ax
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // r14
  unsigned __int8 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  const char *v22; // [rsp+0h] [rbp-60h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  v5 = *(_QWORD *)(a3 - 96);
  if ( *(_QWORD *)(v5 + 8) != *(_QWORD *)(a2 + 8) )
    return 0;
  v6 = *(_QWORD *)(a3 - 64);
  v7 = *(_QWORD *)(a3 - 32);
  v9 = 0;
  v10 = sub_9A18B0(a2, (_BYTE *)v5, *(_QWORD *)(a1 + 88), a4, 0);
  if ( !HIBYTE(v10) )
    return v9;
  if ( !(_BYTE)v10 )
    v6 = v7;
  if ( !v6 )
    return 0;
  v11 = *(_QWORD *)(a2 + 8);
  v23 = 257;
  if ( a4 )
  {
    v13 = v6;
    v12 = sub_AD6450(v11);
  }
  else
  {
    v12 = v6;
    v13 = sub_AD6400(v11);
  }
  v14 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v9 = v14;
  if ( v14 )
  {
    sub_B44260((__int64)v14, *(_QWORD *)(v13 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v9 - 12) )
    {
      v15 = *((_QWORD *)v9 - 11);
      **((_QWORD **)v9 - 10) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *((_QWORD *)v9 - 10);
    }
    *((_QWORD *)v9 - 12) = a2;
    v16 = *(_QWORD *)(a2 + 16);
    *((_QWORD *)v9 - 11) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = v9 - 88;
    *((_QWORD *)v9 - 10) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v9 - 96;
    if ( *((_QWORD *)v9 - 8) )
    {
      v17 = *((_QWORD *)v9 - 7);
      **((_QWORD **)v9 - 6) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *((_QWORD *)v9 - 6);
    }
    *((_QWORD *)v9 - 8) = v13;
    v18 = *(_QWORD *)(v13 + 16);
    *((_QWORD *)v9 - 7) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = v9 - 56;
    *((_QWORD *)v9 - 6) = v13 + 16;
    *(_QWORD *)(v13 + 16) = v9 - 64;
    if ( *((_QWORD *)v9 - 4) )
    {
      v19 = *((_QWORD *)v9 - 3);
      **((_QWORD **)v9 - 2) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *((_QWORD *)v9 - 2);
    }
    *((_QWORD *)v9 - 4) = v12;
    if ( v12 )
    {
      v20 = *(_QWORD *)(v12 + 16);
      *((_QWORD *)v9 - 3) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = v9 - 24;
      *((_QWORD *)v9 - 2) = v12 + 16;
      *(_QWORD *)(v12 + 16) = v9 - 32;
    }
    sub_BD6B50(v9, &v22);
  }
  return v9;
}
