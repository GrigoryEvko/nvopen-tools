// Function: sub_325D810
// Address: 0x325d810
//
__int64 __fastcall sub_325D810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned __int64 v5; // rdx
  int v6; // r9d
  __int64 v7; // r10
  __int64 v8; // r14
  unsigned __int64 v9; // r11
  unsigned __int64 v10; // r15
  __int128 v11; // rax
  __int128 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // r10
  __int64 v16; // rdx
  int v17; // r9d
  __int128 v18; // rax
  int v19; // r9d
  unsigned int v20; // edx
  __int128 v22; // [rsp-20h] [rbp-A0h]
  __int128 v23; // [rsp-20h] [rbp-A0h]
  __int128 v24; // [rsp-20h] [rbp-A0h]
  __int128 v25; // [rsp-20h] [rbp-A0h]
  __int128 v26; // [rsp-10h] [rbp-90h]
  __int128 v27; // [rsp-10h] [rbp-90h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int128 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]

  v4 = a2;
  if ( **(_QWORD **)a1 || **(_QWORD **)(a1 + 8) )
  {
    v7 = sub_34015B0(
           **(_QWORD **)(a1 + 16),
           *(_QWORD *)(a1 + 24),
           **(unsigned int **)(a1 + 32),
           *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
           0,
           0);
    v8 = v7;
    v9 = v5;
    v10 = v5;
    if ( **(_QWORD **)a1 )
    {
      *(_QWORD *)&v29 = v7;
      *((_QWORD *)&v29 + 1) = v5;
      *((_QWORD *)&v22 + 1) = v5;
      *(_QWORD *)&v22 = v7;
      *(_QWORD *)&v11 = sub_3406EB0(
                          **(_QWORD **)(a1 + 16),
                          192,
                          *(_QWORD *)(a1 + 24),
                          **(_DWORD **)(a1 + 32),
                          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                          v6,
                          v22,
                          *(_OWORD *)*(_QWORD *)(a1 + 40));
      v28 = **(_QWORD **)(a1 + 16);
      *(_QWORD *)&v12 = sub_3406EB0(
                          v28,
                          187,
                          *(_QWORD *)(a1 + 24),
                          **(_DWORD **)(a1 + 32),
                          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                          DWORD2(v11),
                          *(_OWORD *)*(_QWORD *)a1,
                          v11);
      v13 = sub_3406EB0(
              v28,
              186,
              *(_QWORD *)(a1 + 24),
              **(_DWORD **)(a1 + 32),
              *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
              DWORD2(v12),
              v29,
              v12);
      v9 = *((_QWORD *)&v29 + 1);
      v7 = v29;
      v8 = v13;
      v10 = v14 | v10 & 0xFFFFFFFF00000000LL;
    }
    if ( **(_QWORD **)(a1 + 8) )
    {
      *((_QWORD *)&v23 + 1) = v9;
      *(_QWORD *)&v23 = v7;
      v15 = sub_3406EB0(
              **(_QWORD **)(a1 + 16),
              190,
              *(_QWORD *)(a1 + 24),
              **(_DWORD **)(a1 + 32),
              *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
              v6,
              v23,
              *(_OWORD *)*(_QWORD *)(a1 + 48));
      *((_QWORD *)&v26 + 1) = v16;
      *(_QWORD *)&v26 = v15;
      v30 = **(_QWORD **)(a1 + 16);
      *(_QWORD *)&v18 = sub_3406EB0(
                          v30,
                          187,
                          *(_QWORD *)(a1 + 24),
                          **(_DWORD **)(a1 + 32),
                          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                          v17,
                          *(_OWORD *)*(_QWORD *)(a1 + 8),
                          v26);
      *((_QWORD *)&v24 + 1) = v10;
      *(_QWORD *)&v24 = v8;
      v8 = sub_3406EB0(
             v30,
             186,
             *(_QWORD *)(a1 + 24),
             **(_DWORD **)(a1 + 32),
             *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
             v19,
             v24,
             v18);
      v10 = v20 | v10 & 0xFFFFFFFF00000000LL;
    }
    *((_QWORD *)&v27 + 1) = v10;
    *(_QWORD *)&v27 = v8;
    *((_QWORD *)&v25 + 1) = a3;
    *(_QWORD *)&v25 = a2;
    return sub_3406EB0(
             **(_QWORD **)(a1 + 16),
             186,
             *(_QWORD *)(a1 + 24),
             **(_DWORD **)(a1 + 32),
             *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
             v6,
             v25,
             v27);
  }
  return v4;
}
