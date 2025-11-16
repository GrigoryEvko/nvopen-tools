// Function: sub_1179150
// Address: 0x1179150
//
__int64 __fastcall sub_1179150(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // r13
  unsigned int **v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // r14
  unsigned __int8 *v11; // r12
  __int64 *v13; // rax
  __int64 v14; // rbx
  unsigned __int8 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  const char *v22[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  v5 = a2;
  v8 = *(unsigned int ***)(*(_QWORD *)a1 + 32LL);
  v9 = *(__int64 **)(a1 + 8);
  v23 = 257;
  v10 = sub_B36550(v8, a3, *v9, a4, (__int64)v22, 0);
  if ( a5 )
  {
    v5 = v10;
    v10 = a2;
  }
  if ( **(_BYTE **)(a1 + 16) || **(_BYTE **)(a1 + 24) && **(_QWORD **)(a1 + 32) == v5 )
  {
    v13 = *(__int64 **)(a1 + 40);
    v23 = 257;
    v14 = *v13;
    v15 = (unsigned __int8 *)sub_BD2C40(72, 3u);
    v11 = v15;
    if ( v15 )
    {
      sub_B44260((__int64)v15, *(_QWORD *)(v10 + 8), 57, 3u, 0, 0);
      if ( *((_QWORD *)v11 - 12) )
      {
        v16 = *((_QWORD *)v11 - 11);
        **((_QWORD **)v11 - 10) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *((_QWORD *)v11 - 10);
      }
      *((_QWORD *)v11 - 12) = v5;
      if ( v5 )
      {
        v17 = *(_QWORD *)(v5 + 16);
        *((_QWORD *)v11 - 11) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = v11 - 88;
        *((_QWORD *)v11 - 10) = v5 + 16;
        *(_QWORD *)(v5 + 16) = v11 - 96;
      }
      if ( *((_QWORD *)v11 - 8) )
      {
        v18 = *((_QWORD *)v11 - 7);
        **((_QWORD **)v11 - 6) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *((_QWORD *)v11 - 6);
      }
      *((_QWORD *)v11 - 8) = v10;
      v19 = *(_QWORD *)(v10 + 16);
      *((_QWORD *)v11 - 7) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = v11 - 56;
      *((_QWORD *)v11 - 6) = v10 + 16;
      *(_QWORD *)(v10 + 16) = v11 - 64;
      if ( *((_QWORD *)v11 - 4) )
      {
        v20 = *((_QWORD *)v11 - 3);
        **((_QWORD **)v11 - 2) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *((_QWORD *)v11 - 2);
      }
      *((_QWORD *)v11 - 4) = v14;
      if ( v14 )
      {
        v21 = *(_QWORD *)(v14 + 16);
        *((_QWORD *)v11 - 3) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = v11 - 24;
        *((_QWORD *)v11 - 2) = v14 + 16;
        *(_QWORD *)(v14 + 16) = v11 - 32;
      }
      sub_BD6B50(v11, v22);
    }
  }
  else
  {
    v23 = 257;
    return sub_B504D0(28, v5, v10, (__int64)v22, 0, 0);
  }
  return (__int64)v11;
}
