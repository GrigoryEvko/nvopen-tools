// Function: sub_33EC010
// Address: 0x33ec010
//
__int64 *__fastcall sub_33EC010(
        _QWORD *a1,
        __int64 *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  int v6; // r15d
  __int64 v9; // rax
  __int64 *result; // rax
  int v11; // r9d
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  unsigned int v20; // [rsp-10h] [rbp-70h]
  int v21; // [rsp+8h] [rbp-58h]
  __int64 *v22; // [rsp+28h] [rbp-38h] BYREF

  v6 = a4;
  v9 = a2[5];
  if ( *(_QWORD *)v9 != a3
    || *(_DWORD *)(v9 + 8) != (_DWORD)a4
    || *(_QWORD *)(v9 + 40) != a5
    || *(_DWORD *)(v9 + 48) != (_DWORD)a6 )
  {
    v22 = 0;
    v21 = a6;
    result = sub_33E33B0((__int64)a1, (__int64)a2, a3, a4, a5, a6, (__int64 *)&v22);
    if ( result )
      return result;
    v11 = v21;
    if ( v22 && (v12 = sub_33EB970((__int64)a1, (__int64)a2, v20), v11 = v21, !v12) )
    {
      v13 = a2[5];
      v22 = 0;
      v14 = *(_QWORD *)v13;
      if ( a3 != *(_QWORD *)v13 )
        goto LABEL_6;
    }
    else
    {
      v13 = a2[5];
      v14 = *(_QWORD *)v13;
      if ( a3 != *(_QWORD *)v13 )
        goto LABEL_6;
    }
    if ( *(_DWORD *)(v13 + 8) == v6 )
    {
LABEL_14:
      v17 = *(_QWORD *)(v13 + 40);
      if ( v17 != a5 || *(_DWORD *)(v13 + 48) != v11 )
      {
        if ( v17 )
        {
          v18 = *(_QWORD *)(v13 + 72);
          **(_QWORD **)(v13 + 64) = v18;
          if ( v18 )
            *(_QWORD *)(v18 + 24) = *(_QWORD *)(v13 + 64);
        }
        *(_QWORD *)(v13 + 40) = a5;
        *(_DWORD *)(v13 + 48) = v11;
        if ( a5 )
        {
          v19 = *(_QWORD *)(a5 + 56);
          *(_QWORD *)(v13 + 72) = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 24) = v13 + 72;
          *(_QWORD *)(v13 + 64) = a5 + 56;
          *(_QWORD *)(a5 + 56) = v13 + 40;
        }
      }
      sub_33CEF80(a1, (__int64)a2);
      if ( v22 )
        sub_C657C0(a1 + 65, a2, v22, (__int64)off_4A367D0);
      return a2;
    }
LABEL_6:
    if ( v14 )
    {
      v15 = *(_QWORD *)(v13 + 32);
      **(_QWORD **)(v13 + 24) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 24) = *(_QWORD *)(v13 + 24);
    }
    *(_QWORD *)v13 = a3;
    *(_DWORD *)(v13 + 8) = v6;
    if ( a3 )
    {
      v16 = *(_QWORD *)(a3 + 56);
      *(_QWORD *)(v13 + 32) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 24) = v13 + 32;
      *(_QWORD *)(v13 + 24) = a3 + 56;
      *(_QWORD *)(a3 + 56) = v13;
    }
    v13 = a2[5];
    goto LABEL_14;
  }
  return a2;
}
