// Function: sub_1D44C70
// Address: 0x1d44c70
//
__int64 __fastcall sub_1D44C70(__int64 a1, __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // r12
  char v12; // di
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v19; // [rsp+18h] [rbp-88h]
  __int64 v21; // [rsp+30h] [rbp-70h] BYREF
  __int64 v22; // [rsp+38h] [rbp-68h] BYREF
  __int64 (__fastcall **v23)(); // [rsp+40h] [rbp-60h] BYREF
  __int64 v24; // [rsp+48h] [rbp-58h]
  __int64 v25; // [rsp+50h] [rbp-50h]
  __int64 *v26; // [rsp+58h] [rbp-48h]
  __int64 *v27; // [rsp+60h] [rbp-40h]

  if ( a2 != a4 )
  {
    if ( *(_DWORD *)(a2 + 60) != 1 )
      goto LABEL_3;
    return sub_1D44850(a1, a2, a3, a4, a5);
  }
  result = a5;
  if ( a3 == a5 )
    return result;
  if ( *(_DWORD *)(a2 + 60) == 1 )
    return sub_1D44850(a1, a2, a3, a4, a5);
LABEL_3:
  sub_1D306C0(a1, a2, a3, a4, a5, 0, 0, 1);
  v8 = *(_QWORD *)(a1 + 664);
  v9 = *(_QWORD *)(a2 + 48);
  v22 = 0;
  v25 = a1;
  v24 = v8;
  *(_QWORD *)(a1 + 664) = &v23;
  v23 = off_49F99D8;
  v26 = &v21;
  v27 = &v22;
  v10 = 0;
  v21 = v9;
LABEL_4:
  if ( v9 != v10 )
  {
    do
    {
      v11 = *(__int64 **)(v9 + 16);
      v12 = 0;
      do
      {
LABEL_8:
        if ( *(_DWORD *)(v9 + 8) == a3 )
        {
          v13 = v9;
          if ( !v12 )
          {
            v19 = v9;
            sub_1D2D480(a1, (__int64)v11, v9);
            v13 = v21;
            v9 = v19;
          }
          while ( 1 )
          {
            v21 = *(_QWORD *)(v13 + 32);
            if ( *(_QWORD *)v9 )
            {
              v14 = *(_QWORD *)(v9 + 32);
              **(_QWORD **)(v9 + 24) = v14;
              if ( v14 )
                *(_QWORD *)(v14 + 24) = *(_QWORD *)(v9 + 24);
            }
            *(_QWORD *)v9 = a4;
            *(_DWORD *)(v9 + 8) = a5;
            if ( a4 )
            {
              v15 = *(_QWORD *)(a4 + 48);
              *(_QWORD *)(v9 + 32) = v15;
              if ( v15 )
                *(_QWORD *)(v15 + 24) = v9 + 32;
              *(_QWORD *)(v9 + 24) = a4 + 48;
              *(_QWORD *)(a4 + 48) = v9;
            }
            if ( ((*(_BYTE *)(a2 + 26) & 4) != 0) != ((*(_BYTE *)(a4 + 26) & 4) != 0) )
              sub_1D18440((_QWORD *)a1, (__int64)v11);
            v13 = v21;
            if ( v21 == v22 || v11 != *(__int64 **)(v21 + 16) )
              goto LABEL_26;
            if ( a3 != *(_DWORD *)(v21 + 8) )
            {
              v9 = *(_QWORD *)(v21 + 32);
              v10 = v22;
              v21 = v9;
              if ( v22 == v9 )
                goto LABEL_26;
              v12 = 1;
              if ( v11 == *(__int64 **)(v9 + 16) )
                goto LABEL_8;
              goto LABEL_25;
            }
            v9 = v21;
          }
        }
        v9 = *(_QWORD *)(v9 + 32);
        v10 = v22;
        v21 = v9;
      }
      while ( v9 != v22 && v11 == *(__int64 **)(v9 + 16) );
LABEL_25:
      if ( !v12 )
        goto LABEL_4;
LABEL_26:
      sub_1D446C0(a1, v11);
      v9 = v21;
    }
    while ( v21 != v22 );
  }
  *(_DWORD *)(a4 + 64) = *(_DWORD *)(a2 + 64);
  if ( a2 == *(_QWORD *)(a1 + 176) && *(_DWORD *)(a1 + 184) == a3 )
  {
    nullsub_686();
    *(_QWORD *)(a1 + 176) = a4;
    *(_DWORD *)(a1 + 184) = a5;
    sub_1D23870();
  }
  result = v25;
  *(_QWORD *)(v25 + 664) = v24;
  return result;
}
