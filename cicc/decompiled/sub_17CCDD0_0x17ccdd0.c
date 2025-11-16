// Function: sub_17CCDD0
// Address: 0x17ccdd0
//
__int64 __fastcall sub_17CCDD0(__int128 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rcx
  int v6; // r8d
  int v7; // r9d
  unsigned __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v16; // r13
  __int64 v17; // r13
  __int64 v18; // r15
  __int64 v19; // r8
  int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-68h]
  _BYTE *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  _BYTE v27[80]; // [rsp+20h] [rbp-50h] BYREF

  v3 = *(_BYTE *)(a1 + 8);
  if ( v3 != 16 && v3 != 11 )
  {
    if ( v3 == 14 )
    {
      v4 = sub_17CCDD0(*(_QWORD *)(a1 + 24));
      v25 = v27;
      *((_QWORD *)&a1 + 1) = v27;
      v8 = *(_QWORD *)(a1 + 32);
      v9 = v4;
      v26 = 0x400000000LL;
      if ( v8 > 4 )
      {
        sub_16CD150((__int64)&v25, v27, v8, 8, v6, v7);
        *((_QWORD *)&a1 + 1) = v25;
      }
      v10 = (unsigned int)v8;
      LODWORD(v26) = v8;
      v11 = *((_QWORD *)&a1 + 1) + 8LL * (unsigned int)v8;
      if ( v11 != *((_QWORD *)&a1 + 1) )
      {
        do
        {
          **((_QWORD **)&a1 + 1) = v9;
          *((_QWORD *)&a1 + 1) += 8LL;
        }
        while ( v11 != *((_QWORD *)&a1 + 1) );
        *((_QWORD *)&a1 + 1) = v25;
        v10 = (unsigned int)v26;
      }
      v12 = sub_159DFD0(a1, v10, v5);
      v13 = (unsigned __int64)v25;
      v14 = v12;
      if ( v25 == v27 )
        return v14;
    }
    else
    {
      v16 = *(unsigned int *)(a1 + 12);
      v25 = v27;
      v26 = 0x400000000LL;
      if ( (_DWORD)v16 )
      {
        v17 = 8 * v16;
        v18 = 0;
        do
        {
          v19 = sub_17CCDD0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + v18));
          v21 = (unsigned int)v26;
          if ( (unsigned int)v26 >= HIDWORD(v26) )
          {
            v24 = v19;
            sub_16CD150((__int64)&v25, v27, 0, 8, v19, v20);
            v21 = (unsigned int)v26;
            v19 = v24;
          }
          v18 += 8;
          *(_QWORD *)&v25[8 * v21] = v19;
          v22 = (unsigned int)(v26 + 1);
          LODWORD(v26) = v26 + 1;
        }
        while ( v17 != v18 );
        *((_QWORD *)&a1 + 1) = v25;
      }
      else
      {
        v22 = 0;
        *((_QWORD *)&a1 + 1) = v27;
      }
      v23 = sub_159F090((__int64 **)a1, *((__int64 **)&a1 + 1), v22, a3);
      v13 = (unsigned __int64)v25;
      v14 = v23;
      if ( v25 == v27 )
        return v14;
    }
    _libc_free(v13);
    return v14;
  }
  return sub_15A04A0((_QWORD **)a1);
}
