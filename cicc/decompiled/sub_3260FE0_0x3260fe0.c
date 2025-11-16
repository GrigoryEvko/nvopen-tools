// Function: sub_3260FE0
// Address: 0x3260fe0
//
__int64 __fastcall sub_3260FE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 i; // rbx
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // r14
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  int v22; // [rsp+10h] [rbp-40h]

  v8 = *(_QWORD *)(a3 + 56);
  v22 = a6;
  if ( v8 )
  {
    if ( !*(_QWORD *)(v8 + 32) )
    {
      v19 = a1[1];
      v20 = *(__int64 (**)())(*(_QWORD *)v19 + 1256LL);
      if ( v20 == sub_2FE33A0
        || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v20)(v19, a3, a4, a5, a6) )
      {
        return 1;
      }
    }
  }
  v9 = *(_QWORD *)(a2 + 56);
  if ( v9 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v9 + 16);
      if ( *(_DWORD *)(v10 + 24) == 56 )
      {
        if ( (unsigned __int8)sub_33E2390(
                                *a1,
                                *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                                1) )
          return 1;
        v11 = *(_QWORD *)(v10 + 56);
        if ( v11 )
          break;
      }
LABEL_4:
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        goto LABEL_13;
    }
    while ( 1 )
    {
      v12 = *(_QWORD *)(v11 + 16);
      if ( *(_DWORD *)(v12 + 24) == 56 )
      {
        if ( (unsigned __int8)sub_33E2390(
                                *a1,
                                *(_QWORD *)(*(_QWORD *)(v12 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v12 + 40) + 48LL),
                                1) )
          return 1;
      }
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        goto LABEL_4;
    }
  }
LABEL_13:
  for ( i = *(_QWORD *)(a5 + 56); i; i = *(_QWORD *)(i + 32) )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(i + 16);
      if ( a2 != v15 && *(_DWORD *)(v15 + 24) == 58 )
      {
        v16 = *(__int64 **)(v15 + 40);
        v17 = **(_QWORD **)(a3 + 40);
        v18 = *v16;
        if ( a5 == *v16 && *((_DWORD *)v16 + 2) == v22 )
          v18 = v16[5];
        if ( v18 == v17 )
          return 1;
        if ( *(_DWORD *)(v18 + 24) == 56 )
        {
          v21 = v18;
          if ( (unsigned __int8)sub_33E2390(
                                  *a1,
                                  *(_QWORD *)(*(_QWORD *)(v18 + 40) + 40LL),
                                  *(_QWORD *)(*(_QWORD *)(v18 + 40) + 48LL),
                                  1) )
            break;
        }
      }
      i = *(_QWORD *)(i + 32);
      if ( !i )
        return 0;
    }
    if ( v17 == **(_QWORD **)(v21 + 40) )
      return 1;
  }
  return 0;
}
