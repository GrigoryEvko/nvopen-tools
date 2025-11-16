// Function: sub_38CE940
// Address: 0x38ce940
//
__int64 __fastcall sub_38CE940(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 *a9)
{
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 (*v14)(); // rax
  unsigned int v15; // r8d
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20[2]; // [rsp+28h] [rbp-58h] BYREF
  __int64 v21; // [rsp+38h] [rbp-48h] BYREF
  __int64 v22; // [rsp+40h] [rbp-40h] BYREF
  __int64 v23[7]; // [rsp+48h] [rbp-38h] BYREF

  v10 = a5[1];
  v11 = *a5;
  v20[0] = a6;
  v22 = v10;
  v12 = a8 + a5[2];
  v21 = v11;
  v23[0] = v12;
  if ( a1 )
  {
    if ( a4 || (v14 = *(__int64 (**)())(**(_QWORD **)(a1 + 8) + 72LL), v14 == sub_38CB1C0) || !(unsigned __int8)v14() )
    {
      sub_38CE540(a1, a2, a3, a4, &v21, &v22, v23);
      sub_38CE540(a1, a2, a3, a4, &v21, &a7, v23);
      sub_38CE540(a1, a2, a3, a4, v20, &v22, v23);
      sub_38CE540(a1, a2, a3, a4, v20, &a7, v23);
      v11 = v21;
    }
    else
    {
      v11 = v21;
    }
  }
  if ( !v11 )
  {
    v17 = v22;
    if ( v22 )
    {
      v15 = 0;
      if ( a7 )
        return v15;
      v11 = v20[0];
      goto LABEL_10;
    }
    v11 = v20[0];
    goto LABEL_17;
  }
  v15 = 0;
  if ( !v20[0] )
  {
    v17 = v22;
    if ( v22 )
    {
      if ( a7 )
        return v15;
LABEL_10:
      v18 = a9;
      v15 = 1;
      *a9 = v11;
      v19 = v23[0];
      v18[1] = v17;
      v18[2] = v19;
      *((_DWORD *)v18 + 6) = 0;
      return v15;
    }
LABEL_17:
    v17 = a7;
    goto LABEL_10;
  }
  return v15;
}
