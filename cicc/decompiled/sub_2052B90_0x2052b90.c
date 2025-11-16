// Function: sub_2052B90
// Address: 0x2052b90
//
void __fastcall sub_2052B90(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 *v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rsi
  int v17; // edx
  __int64 v18; // rbx
  int v19; // r13d
  __int128 v20; // [rsp-10h] [rbp-70h]
  __int64 *v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+30h] [rbp-30h] BYREF
  int v24; // [rsp+38h] [rbp-28h]

  v5 = sub_15E38F0(**(_QWORD **)(a1 + 712));
  v6 = sub_14DD7D0(v5);
  v10 = (unsigned int)(v6 - 7);
  if ( (unsigned int)v10 > 1 )
  {
    v7 = (unsigned int)(v6 - 9);
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL);
    *(_BYTE *)(v10 + 182) = 1;
    if ( (unsigned int)v7 > 1 )
    {
      if ( v6 == 12 )
        return;
    }
    else
    {
      *(_BYTE *)(v10 + 183) = 1;
    }
  }
  v11 = *(_QWORD *)(a1 + 552);
  v23 = 0;
  v13 = sub_2051DF0((__int64 *)a1, a3, a4, a5, a2, v10, v7, v8, v9);
  v14 = v12;
  v15 = *(_QWORD *)a1;
  v24 = *(_DWORD *)(a1 + 536);
  if ( v15 )
  {
    if ( &v23 != (__int64 *)(v15 + 48) )
    {
      v16 = *(_QWORD *)(v15 + 48);
      v23 = v16;
      if ( v16 )
      {
        v21 = v13;
        v22 = v12;
        sub_1623A60((__int64)&v23, v16, 2);
        v13 = v21;
        v14 = v22;
      }
    }
  }
  *((_QWORD *)&v20 + 1) = v14;
  *(_QWORD *)&v20 = v13;
  v18 = sub_1D309E0((__int64 *)v11, 196, (__int64)&v23, 1, 0, 0, a3, a4, *(double *)a5.m128i_i64, v20);
  v19 = v17;
  if ( v18 )
  {
    nullsub_686();
    *(_QWORD *)(v11 + 176) = v18;
    *(_DWORD *)(v11 + 184) = v19;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v11 + 176) = 0;
    *(_DWORD *)(v11 + 184) = v17;
  }
  if ( v23 )
    sub_161E7C0((__int64)&v23, v23);
}
