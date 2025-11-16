// Function: sub_29B8380
// Address: 0x29b8380
//
__int64 __fastcall sub_29B8380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v11; // rdx
  __int64 *v12; // r8
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // rsi
  __int64 v16; // rcx
  __int64 *i; // rdx
  __int64 v18; // rcx
  __int64 *j; // rdx
  __int64 v20; // rcx
  __int64 v21; // r10
  double v22; // xmm3_8
  __int64 *v23; // rax
  double v24; // xmm3_8
  __int64 v26; // r11
  __int64 v27; // r9
  __int64 *v29; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v30; // [rsp+18h] [rbp-58h]
  __int64 *v31; // [rsp+20h] [rbp-50h]
  __int64 *v32; // [rsp+28h] [rbp-48h]
  __int64 *v33; // [rsp+30h] [rbp-40h]
  __int64 *v34; // [rsp+38h] [rbp-38h]

  sub_29B78C0((__int64 *)&v29, (__int64 *)(a2 + 32), (__int64 *)(a3 + 32), a5, a6);
  v11 = v29;
  if ( ***(_QWORD ***)(a2 + 32) && ***(_QWORD ***)(a3 + 32) || !*(_QWORD *)*v29 )
  {
    v12 = v30;
    v13 = v32;
    v14 = 0;
    v15 = v34;
    if ( v30 != v29 )
    {
      do
      {
        v16 = *v11++;
        *(_QWORD *)(v16 + 40) = v14;
        v14 += *(_QWORD *)(v16 + 16);
      }
      while ( v11 != v12 );
    }
    for ( i = v31; v13 != i; v14 += *(_QWORD *)(v18 + 16) )
    {
      v18 = *i++;
      *(_QWORD *)(v18 + 40) = v14;
    }
    for ( j = v33; v15 != j; v14 += *(_QWORD *)(v20 + 16) )
    {
      v20 = *j++;
      *(_QWORD *)(v20 + 40) = v14;
    }
    v21 = 0;
    v22 = 0.0;
    do
    {
      v23 = *(__int64 **)(a4 + v21);
      if ( v23 )
      {
        v26 = *v23;
        if ( *v23 != v23[1] )
        {
          do
            v22 = v22
                + sub_29B8240(
                    *(_QWORD *)(**(_QWORD **)v26 + 40LL),
                    *(_QWORD *)(**(_QWORD **)v26 + 16LL),
                    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v26 + 8LL) + 40LL),
                    *(_QWORD *)(*(_QWORD *)v26 + 16LL),
                    *(_BYTE *)(*(_QWORD *)v26 + 24LL));
          while ( v27 != v26 );
        }
      }
      v21 += 8;
    }
    while ( v21 != 16 );
    v24 = v22 - *(double *)(a2 + 8);
    *(_QWORD *)(a1 + 8) = a5;
    *(_DWORD *)(a1 + 16) = a6;
    *(double *)a1 = v24;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)a1 = 0xBFF0000000000000LL;
  }
  return a1;
}
