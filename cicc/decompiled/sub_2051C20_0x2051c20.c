// Function: sub_2051C20
// Address: 0x2051c20
//
__int64 *__fastcall sub_2051C20(__int64 *a1, double a2, double a3, __m128i a4)
{
  unsigned int v4; // eax
  __int64 v5; // r14
  __int64 *v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rsi
  __int64 *v13; // r12
  int v14; // edx
  int v15; // r13d
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // r13
  __int128 v19; // [rsp-10h] [rbp-90h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  int v22; // [rsp+48h] [rbp-38h]

  v4 = *((_DWORD *)a1 + 28);
  v5 = a1[69];
  if ( !v4 )
    return *(__int64 **)(v5 + 176);
  v7 = (__int64 *)a1[13];
  if ( v4 == 1 )
  {
    v17 = *v7;
    v18 = v7[1];
    if ( *v7 )
    {
      nullsub_686();
      *(_QWORD *)(v5 + 176) = v17;
      *(_DWORD *)(v5 + 184) = v18;
      sub_1D23870();
    }
    else
    {
      v20 = v7[1];
      *(_QWORD *)(v5 + 176) = 0;
      *(_DWORD *)(v5 + 184) = v20;
    }
    *((_DWORD *)a1 + 28) = 0;
    return (__int64 *)v17;
  }
  else
  {
    v8 = a1[13];
    v9 = v4;
    v10 = *a1;
    v11 = *((_DWORD *)a1 + 134);
    v21 = 0;
    v22 = v11;
    if ( v10 )
    {
      if ( &v21 != (__int64 *)(v10 + 48) )
      {
        v12 = *(_QWORD *)(v10 + 48);
        v21 = v12;
        if ( v12 )
          sub_1623A60((__int64)&v21, v12, 2);
      }
    }
    *((_QWORD *)&v19 + 1) = v9;
    *(_QWORD *)&v19 = v8;
    v13 = sub_1D359D0((__int64 *)v5, 2, (__int64)&v21, 1, 0, 0, a2, a3, a4, v19);
    v15 = v14;
    if ( v21 )
      sub_161E7C0((__int64)&v21, v21);
    *((_DWORD *)a1 + 28) = 0;
    v16 = a1[69];
    if ( v13 )
    {
      nullsub_686();
      *(_QWORD *)(v16 + 176) = v13;
      *(_DWORD *)(v16 + 184) = v15;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(v16 + 176) = 0;
      *(_DWORD *)(v16 + 184) = v15;
    }
    return v13;
  }
}
