// Function: sub_3415F70
// Address: 0x3415f70
//
__int64 __fastcall sub_3415F70(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v6; // r14
  int v7; // r15d
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  char v19; // si
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 result; // rax
  __int64 *v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rbx
  int v29; // [rsp+Ch] [rbp-94h]
  __int64 v30; // [rsp+30h] [rbp-70h] BYREF
  __int64 v31; // [rsp+38h] [rbp-68h] BYREF
  __int64 (__fastcall **v32)(); // [rsp+40h] [rbp-60h] BYREF
  __int64 v33; // [rsp+48h] [rbp-58h]
  __int64 v34; // [rsp+50h] [rbp-50h]
  __int64 *v35; // [rsp+58h] [rbp-48h]
  __int64 *v36; // [rsp+60h] [rbp-40h]

  v29 = *(_DWORD *)(a2 + 68);
  if ( v29 == 1 )
    return sub_3415D80(a1, a2, 0, *a3, a3[1]);
  v6 = a3;
  v7 = 0;
  if ( v29 )
  {
    do
    {
      v8 = *v6;
      v9 = v6[1];
      v10 = v7++;
      v6 += 2;
      sub_33F9B80(a1, a2, v10, v8, v9, 0, 0, 1);
      sub_34151B0(a1, a2, *(v6 - 2), v11, v12, v13);
    }
    while ( v7 != v29 );
  }
  v14 = *(_QWORD *)(a1 + 768);
  v15 = *(_QWORD *)(a2 + 56);
  v34 = a1;
  v31 = 0;
  v33 = v14;
  *(_QWORD *)(a1 + 768) = &v32;
  v35 = &v30;
  v16 = &v31;
  v30 = v15;
  v32 = off_4A36748;
  v36 = &v31;
  if ( v15 )
  {
    do
    {
      v17 = *(_QWORD *)(v15 + 16);
      sub_33EB970(a1, v17, (unsigned int)v16);
      v18 = v30;
      v19 = 0;
      do
      {
        v20 = *(_QWORD *)(v18 + 32);
        v21 = 2LL * *(unsigned int *)(v18 + 8);
        v30 = v20;
        v22 = &a3[v21];
        if ( *(_QWORD *)v18 )
        {
          **(_QWORD **)(v18 + 24) = v20;
          if ( v20 )
            *(_QWORD *)(v20 + 24) = *(_QWORD *)(v18 + 24);
        }
        *(_QWORD *)v18 = *v22;
        *(_DWORD *)(v18 + 8) = *((_DWORD *)v22 + 2);
        v23 = *v22;
        if ( *v22 )
        {
          v24 = *(_QWORD *)(v23 + 56);
          *(_QWORD *)(v18 + 32) = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 24) = v18 + 32;
          *(_QWORD *)(v18 + 24) = v23 + 56;
          *(_QWORD *)(v23 + 56) = v18;
          v23 = *v22;
        }
        v19 |= (*(_BYTE *)(v23 + 32) & 4) != 0;
        v18 = v30;
      }
      while ( v31 != v30 && v17 == *(_QWORD *)(v30 + 16) );
      if ( v19 != ((*(_BYTE *)(a2 + 32) & 4) != 0) )
        sub_33CEF80((_QWORD *)a1, v17);
      sub_3415B20(a1, v17);
      v15 = v30;
    }
    while ( v31 != v30 );
  }
  if ( a2 == *(_QWORD *)(a1 + 384) )
  {
    v26 = &a3[2 * *(unsigned int *)(a1 + 392)];
    v27 = *v26;
    v28 = v26[1];
    if ( v27 )
    {
      nullsub_1875();
      *(_QWORD *)(a1 + 384) = v27;
      *(_DWORD *)(a1 + 392) = v28;
      sub_33E2B60();
    }
    else
    {
      *(_QWORD *)(a1 + 384) = 0;
      *(_DWORD *)(a1 + 392) = v28;
    }
  }
  result = v34;
  *(_QWORD *)(v34 + 768) = v33;
  return result;
}
