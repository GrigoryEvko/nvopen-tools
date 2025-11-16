// Function: sub_E817B0
// Address: 0xe817b0
//
__int64 __fastcall sub_E817B0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // [rsp+28h] [rbp-58h] BYREF
  __int64 v18; // [rsp+30h] [rbp-50h] BYREF
  __int64 v19; // [rsp+38h] [rbp-48h] BYREF
  __int64 v20; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v21[7]; // [rsp+48h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a4 + 8);
  v9 = *(_QWORD *)a4;
  v19 = *(_QWORD *)a5;
  v10 = *(_QWORD *)(a5 + 8);
  v18 = v8;
  v11 = *(_QWORD *)(a4 + 16);
  v20 = v10;
  LODWORD(v10) = *(_DWORD *)(a5 + 24);
  v17 = v9;
  if ( *(_DWORD *)(a4 + 24) != (_DWORD)v10 )
    return 0;
  v21[0] = *(_QWORD *)(a5 + 16) + v11;
  if ( a1 )
  {
    sub_E81190(a1, a2, a3, &v17, &v18, v21);
    sub_E81190(a1, a2, a3, &v17, &v20, v21);
    sub_E81190(a1, a2, a3, &v19, &v18, v21);
    sub_E81190(a1, a2, a3, &v19, &v20, v21);
    v9 = v17;
  }
  if ( !v9 )
  {
    v15 = v18;
    if ( v18 )
    {
      if ( v20 )
        return 0;
      v9 = v19;
      goto LABEL_9;
    }
    v9 = v19;
LABEL_14:
    v15 = v20;
    goto LABEL_9;
  }
  if ( v19 )
    return 0;
  v15 = v18;
  if ( !v18 )
    goto LABEL_14;
  if ( v20 )
    return 0;
LABEL_9:
  *(_QWORD *)a6 = v9;
  v16 = v21[0];
  *(_QWORD *)(a6 + 8) = v15;
  *(_QWORD *)(a6 + 16) = v16;
  *(_DWORD *)(a6 + 24) = 0;
  return 1;
}
