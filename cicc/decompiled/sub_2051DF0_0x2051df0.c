// Function: sub_2051DF0
// Address: 0x2051df0
//
__int64 *__fastcall sub_2051DF0(
        __int64 *a1,
        double a2,
        double a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 *v9; // r14
  __int64 v10; // rcx
  __int64 *v11; // r12
  __int64 v12; // r13
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rsi
  int v22; // edx
  int v23; // r13d
  __int64 v24; // r14
  __int64 **v26; // rdi
  __int128 v27; // [rsp-10h] [rbp-90h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+40h] [rbp-40h] BYREF
  int v30; // [rsp+48h] [rbp-38h]

  v9 = (__int64 *)a1[69];
  v10 = *((unsigned int *)a1 + 100);
  v11 = (__int64 *)v9[22];
  v12 = v9[23];
  if ( (_DWORD)v10 )
  {
    v14 = a1[49];
    if ( *((_WORD *)v11 + 12) != 1 )
    {
      v15 = v14;
      while ( 1 )
      {
        v16 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
        if ( v11 == *(__int64 **)v16 && *(_DWORD *)(v16 + 8) == (_DWORD)v12 )
          break;
        v15 += 16;
        if ( v14 + 16LL * (unsigned int)(v10 - 1) + 16 == v15 )
        {
          if ( (unsigned int)v10 >= *((_DWORD *)a1 + 101) )
          {
            sub_16CD150((__int64)(a1 + 49), a1 + 51, 0, 16, v12, a9);
            v14 = a1[49];
            v10 = *((unsigned int *)a1 + 100);
          }
          v26 = (__int64 **)(16 * v10 + v14);
          *v26 = v11;
          v26[1] = (__int64 *)v12;
          v9 = (__int64 *)a1[69];
          v14 = a1[49];
          LODWORD(v10) = *((_DWORD *)a1 + 100) + 1;
          *((_DWORD *)a1 + 100) = v10;
          break;
        }
      }
    }
    v17 = *((_DWORD *)a1 + 134);
    v18 = *a1;
    v19 = v14;
    v20 = (unsigned int)v10;
    v29 = 0;
    v30 = v17;
    if ( v18 )
    {
      if ( &v29 != (__int64 *)(v18 + 48) )
      {
        v21 = *(_QWORD *)(v18 + 48);
        v29 = v21;
        if ( v21 )
        {
          v28 = (unsigned int)v10;
          sub_1623A60((__int64)&v29, v21, 2);
          v19 = v14;
          v20 = v28;
        }
      }
    }
    *((_QWORD *)&v27 + 1) = v20;
    *(_QWORD *)&v27 = v19;
    v11 = sub_1D359D0(v9, 2, (__int64)&v29, 1, 0, 0, a2, a3, a4, v27);
    v23 = v22;
    if ( v29 )
      sub_161E7C0((__int64)&v29, v29);
    *((_DWORD *)a1 + 100) = 0;
    v24 = a1[69];
    if ( v11 )
    {
      nullsub_686();
      *(_QWORD *)(v24 + 176) = v11;
      *(_DWORD *)(v24 + 184) = v23;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(v24 + 176) = 0;
      *(_DWORD *)(v24 + 184) = v23;
    }
  }
  return v11;
}
