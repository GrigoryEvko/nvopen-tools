// Function: sub_1D44850
// Address: 0x1d44850
//
__int64 __fastcall sub_1D44850(__int64 a1, __int64 a2, int a3, __int64 a4, int a5)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // edx
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 result; // rax
  unsigned int v15; // [rsp-10h] [rbp-C0h]
  __int64 v17; // [rsp+40h] [rbp-70h] BYREF
  __int64 v18; // [rsp+48h] [rbp-68h] BYREF
  __int64 (__fastcall **v19)(); // [rsp+50h] [rbp-60h] BYREF
  __int64 v20; // [rsp+58h] [rbp-58h]
  __int64 v21; // [rsp+60h] [rbp-50h]
  __int64 *v22; // [rsp+68h] [rbp-48h]
  __int64 *v23; // [rsp+70h] [rbp-40h]

  sub_1D306C0(a1, a2, a3, a4, a5, 0, 0, 1);
  v7 = *(_QWORD *)(a1 + 664);
  v8 = *(_QWORD *)(a2 + 48);
  v19 = off_49F99D8;
  v20 = v7;
  *(_QWORD *)(a1 + 664) = &v19;
  v22 = &v17;
  v23 = &v18;
  v9 = v15;
  v17 = v8;
  v18 = 0;
  v21 = a1;
  if ( v8 )
  {
    do
    {
      v10 = *(__int64 **)(v8 + 16);
      sub_1D2D480(a1, (__int64)v10, v9);
      v11 = v17;
      do
      {
        v12 = *(_QWORD *)(v11 + 32);
        v17 = v12;
        if ( *(_QWORD *)v11 )
        {
          **(_QWORD **)(v11 + 24) = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 24) = *(_QWORD *)(v11 + 24);
        }
        *(_QWORD *)v11 = a4;
        *(_DWORD *)(v11 + 8) = a5;
        if ( a4 )
        {
          v13 = *(_QWORD *)(a4 + 48);
          *(_QWORD *)(v11 + 32) = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 24) = v11 + 32;
          *(_QWORD *)(v11 + 24) = a4 + 48;
          *(_QWORD *)(a4 + 48) = v11;
        }
        if ( ((*(_BYTE *)(a2 + 26) & 4) != 0) != ((*(_BYTE *)(a4 + 26) & 4) != 0) )
          sub_1D18440((_QWORD *)a1, (__int64)v10);
        v11 = v17;
      }
      while ( v17 != v18 && v10 == *(__int64 **)(v17 + 16) );
      sub_1D446C0(a1, v10);
      v8 = v17;
    }
    while ( v18 != v17 );
  }
  if ( *(_DWORD *)(a1 + 56) | *(_DWORD *)(a2 + 64) )
    *(_DWORD *)(a4 + 64) = *(_DWORD *)(a2 + 64);
  if ( a2 == *(_QWORD *)(a1 + 176) && *(_DWORD *)(a1 + 184) == a3 )
  {
    if ( a4 )
    {
      nullsub_686();
      *(_QWORD *)(a1 + 176) = a4;
      *(_DWORD *)(a1 + 184) = a5;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = a5;
    }
  }
  result = v21;
  *(_QWORD *)(v21 + 664) = v20;
  return result;
}
