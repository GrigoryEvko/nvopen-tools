// Function: sub_1D44A40
// Address: 0x1d44a40
//
__int64 __fastcall sub_1D44A40(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // r15d
  int v5; // r14d
  int v6; // edx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 result; // rax
  __int64 *v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // [rsp+20h] [rbp-70h] BYREF
  __int64 v22; // [rsp+28h] [rbp-68h] BYREF
  __int64 (__fastcall **v23)(); // [rsp+30h] [rbp-60h] BYREF
  __int64 v24; // [rsp+38h] [rbp-58h]
  __int64 v25; // [rsp+40h] [rbp-50h]
  __int64 *v26; // [rsp+48h] [rbp-48h]
  __int64 *v27; // [rsp+50h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 60);
  if ( v4 == 1 )
    return sub_1D44850(a1, a2, 0, *a3, a3[1]);
  v5 = 0;
  if ( v4 )
  {
    do
    {
      v6 = v5++;
      sub_1D306C0(a1, a2, v6, *a3, a3[1], 0, 0, 1);
    }
    while ( v5 != v4 );
  }
  v7 = *(_QWORD *)(a1 + 664);
  v8 = *(_QWORD *)(a2 + 48);
  v25 = a1;
  v22 = 0;
  v24 = v7;
  *(_QWORD *)(a1 + 664) = &v23;
  v26 = &v21;
  v9 = &v22;
  v21 = v8;
  v23 = off_49F99D8;
  v27 = &v22;
  if ( v8 )
  {
    do
    {
      v10 = *(__int64 **)(v8 + 16);
      sub_1D2D480(a1, (__int64)v10, (unsigned int)v9);
      v11 = v21;
      do
      {
        v12 = *(_QWORD *)(v11 + 32);
        v13 = 2LL * *(unsigned int *)(v11 + 8);
        v21 = v12;
        v14 = &a3[v13];
        if ( *(_QWORD *)v11 )
        {
          **(_QWORD **)(v11 + 24) = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 24) = *(_QWORD *)(v11 + 24);
        }
        *(_QWORD *)v11 = *v14;
        *(_DWORD *)(v11 + 8) = *((_DWORD *)v14 + 2);
        v15 = *v14;
        if ( v15 )
        {
          v16 = *(_QWORD *)(v15 + 48);
          *(_QWORD *)(v11 + 32) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 24) = v11 + 32;
          *(_QWORD *)(v11 + 24) = v15 + 48;
          *(_QWORD *)(v15 + 48) = v11;
        }
        if ( ((*(_BYTE *)(a2 + 26) & 4) != 0) != ((*(_BYTE *)(*a3 + 26) & 4) != 0) )
          sub_1D18440((_QWORD *)a1, (__int64)v10);
        v11 = v21;
      }
      while ( v21 != v22 && v10 == *(__int64 **)(v21 + 16) );
      sub_1D446C0(a1, v10);
      v8 = v21;
    }
    while ( v22 != v21 );
  }
  *(_DWORD *)(*a3 + 64) = *(_DWORD *)(a2 + 64);
  if ( a2 == *(_QWORD *)(a1 + 176) )
  {
    v18 = &a3[2 * *(unsigned int *)(a1 + 184)];
    v19 = *v18;
    v20 = v18[1];
    if ( v19 )
    {
      nullsub_686();
      *(_QWORD *)(a1 + 176) = v19;
      *(_DWORD *)(a1 + 184) = v20;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = v20;
    }
  }
  result = v25;
  *(_QWORD *)(v25 + 664) = v24;
  return result;
}
