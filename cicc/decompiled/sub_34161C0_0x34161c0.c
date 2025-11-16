// Function: sub_34161C0
// Address: 0x34161c0
//
__int64 __fastcall sub_34161C0(__int64 a1, __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r12
  char v15; // di
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v22; // [rsp+18h] [rbp-88h]
  __int64 v24; // [rsp+30h] [rbp-70h] BYREF
  __int64 v25; // [rsp+38h] [rbp-68h] BYREF
  __int64 (__fastcall **v26)(); // [rsp+40h] [rbp-60h] BYREF
  __int64 v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 *v29; // [rsp+58h] [rbp-48h]
  __int64 *v30; // [rsp+60h] [rbp-40h]

  if ( a2 != a4 )
  {
    if ( *(_DWORD *)(a2 + 68) != 1 )
      goto LABEL_3;
    return sub_3415D80(a1, a2, a3, a4, a5);
  }
  result = a5;
  if ( a3 == a5 )
    return result;
  if ( *(_DWORD *)(a2 + 68) == 1 )
    return sub_3415D80(a1, a2, a3, a4, a5);
LABEL_3:
  sub_33F9B80(a1, a2, a3, a4, a5, 0, 0, 1);
  sub_34151B0(a1, a2, a4, v8, v9, v10);
  v11 = *(_QWORD *)(a1 + 768);
  v12 = *(_QWORD *)(a2 + 56);
  v25 = 0;
  v28 = a1;
  v27 = v11;
  *(_QWORD *)(a1 + 768) = &v26;
  v26 = off_4A36748;
  v29 = &v24;
  v30 = &v25;
  v13 = 0;
  v24 = v12;
LABEL_4:
  if ( v12 != v13 )
  {
    do
    {
      v14 = *(_QWORD *)(v12 + 16);
      v15 = 0;
      do
      {
LABEL_8:
        if ( *(_DWORD *)(v12 + 8) == a3 )
        {
          v16 = v12;
          if ( !v15 )
          {
            v22 = v12;
            sub_33EB970(a1, v14, v12);
            v16 = v24;
            v12 = v22;
          }
          while ( 1 )
          {
            v24 = *(_QWORD *)(v16 + 32);
            if ( *(_QWORD *)v12 )
            {
              v17 = *(_QWORD *)(v12 + 32);
              **(_QWORD **)(v12 + 24) = v17;
              if ( v17 )
                *(_QWORD *)(v17 + 24) = *(_QWORD *)(v12 + 24);
            }
            *(_QWORD *)v12 = a4;
            *(_DWORD *)(v12 + 8) = a5;
            if ( a4 )
            {
              v18 = *(_QWORD *)(a4 + 56);
              *(_QWORD *)(v12 + 32) = v18;
              if ( v18 )
                *(_QWORD *)(v18 + 24) = v12 + 32;
              *(_QWORD *)(v12 + 24) = a4 + 56;
              *(_QWORD *)(a4 + 56) = v12;
            }
            if ( ((*(_BYTE *)(a2 + 32) & 4) != 0) != ((*(_BYTE *)(a4 + 32) & 4) != 0) )
              sub_33CEF80((_QWORD *)a1, v14);
            v16 = v24;
            if ( v24 == v25 || v14 != *(_QWORD *)(v24 + 16) )
              goto LABEL_26;
            if ( a3 != *(_DWORD *)(v24 + 8) )
            {
              v12 = *(_QWORD *)(v24 + 32);
              v13 = v25;
              v24 = v12;
              if ( v12 == v25 )
                goto LABEL_26;
              v15 = 1;
              if ( v14 == *(_QWORD *)(v12 + 16) )
                goto LABEL_8;
              goto LABEL_25;
            }
            v12 = v24;
          }
        }
        v12 = *(_QWORD *)(v12 + 32);
        v13 = v25;
        v24 = v12;
      }
      while ( v12 != v25 && v14 == *(_QWORD *)(v12 + 16) );
LABEL_25:
      if ( !v15 )
        goto LABEL_4;
LABEL_26:
      sub_3415B20(a1, v14);
      v12 = v24;
    }
    while ( v24 != v25 );
  }
  if ( a2 == *(_QWORD *)(a1 + 384) && *(_DWORD *)(a1 + 392) == a3 )
  {
    if ( a4 )
    {
      nullsub_1875();
      *(_QWORD *)(a1 + 384) = a4;
      *(_DWORD *)(a1 + 392) = a5;
      sub_33E2B60();
    }
    else
    {
      *(_QWORD *)(a1 + 384) = 0;
      *(_DWORD *)(a1 + 392) = a5;
    }
  }
  result = v28;
  *(_QWORD *)(v28 + 768) = v27;
  return result;
}
