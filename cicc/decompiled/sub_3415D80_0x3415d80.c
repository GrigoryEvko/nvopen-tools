// Function: sub_3415D80
// Address: 0x3415d80
//
__int64 __fastcall sub_3415D80(__int64 a1, __int64 a2, int a3, __int64 a4, int a5)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 result; // rax
  unsigned int v18; // [rsp-10h] [rbp-C0h]
  __int64 v20; // [rsp+40h] [rbp-70h] BYREF
  __int64 v21; // [rsp+48h] [rbp-68h] BYREF
  __int64 (__fastcall **v22)(); // [rsp+50h] [rbp-60h] BYREF
  __int64 v23; // [rsp+58h] [rbp-58h]
  __int64 v24; // [rsp+60h] [rbp-50h]
  __int64 *v25; // [rsp+68h] [rbp-48h]
  __int64 *v26; // [rsp+70h] [rbp-40h]

  sub_33F9B80(a1, a2, a3, a4, a5, 0, 0, 1);
  sub_34151B0(a1, a2, a4, v7, v8, v9);
  v10 = *(_QWORD *)(a1 + 768);
  v11 = *(_QWORD *)(a2 + 56);
  v22 = off_4A36748;
  v23 = v10;
  *(_QWORD *)(a1 + 768) = &v22;
  v25 = &v20;
  v26 = &v21;
  v12 = v18;
  v20 = v11;
  v21 = 0;
  v24 = a1;
  if ( v11 )
  {
    do
    {
      v13 = *(_QWORD *)(v11 + 16);
      sub_33EB970(a1, v13, v12);
      v14 = v20;
      do
      {
        v15 = *(_QWORD *)(v14 + 32);
        v20 = v15;
        if ( *(_QWORD *)v14 )
        {
          **(_QWORD **)(v14 + 24) = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 24) = *(_QWORD *)(v14 + 24);
        }
        *(_QWORD *)v14 = a4;
        *(_DWORD *)(v14 + 8) = a5;
        if ( a4 )
        {
          v16 = *(_QWORD *)(a4 + 56);
          *(_QWORD *)(v14 + 32) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 24) = v14 + 32;
          *(_QWORD *)(v14 + 24) = a4 + 56;
          *(_QWORD *)(a4 + 56) = v14;
        }
        if ( ((*(_BYTE *)(a2 + 32) & 4) != 0) != ((*(_BYTE *)(a4 + 32) & 4) != 0) )
          sub_33CEF80((_QWORD *)a1, v13);
        v14 = v20;
      }
      while ( v20 != v21 && v13 == *(_QWORD *)(v20 + 16) );
      sub_3415B20(a1, v13);
      v11 = v20;
    }
    while ( v21 != v20 );
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
  result = v24;
  *(_QWORD *)(v24 + 768) = v23;
  return result;
}
