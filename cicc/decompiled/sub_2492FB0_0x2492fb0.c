// Function: sub_2492FB0
// Address: 0x2492fb0
//
__int64 __fastcall sub_2492FB0(_QWORD **a1, unsigned __int8 *a2)
{
  int v3; // eax
  __int64 v4; // rsi
  __int64 **v5; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  _DWORD *v11; // r15
  __int64 *v12; // rsi
  __int64 *v13; // r13
  __int64 v14; // r12
  int v15; // r13d
  unsigned int v16; // r15d
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  unsigned __int64 v21; // r9
  __int64 v22; // rsi
  __int64 *v23; // rdi
  _QWORD *i; // rbx
  __int64 v25; // [rsp+0h] [rbp-C0h]
  bool v26; // [rsp+1Fh] [rbp-A1h] BYREF
  __int64 v27[4]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v28; // [rsp+40h] [rbp-80h] BYREF
  __int64 v29; // [rsp+48h] [rbp-78h]
  _BYTE v30[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = *a2;
  v4 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)(v3 - 12) > 1 )
  {
    if ( (_BYTE)v3 == 18 )
    {
      v7 = sub_2491640(*a1, v4);
      v11 = (_DWORD *)sub_BCAC60(v7, v4, v8, v9, v10);
      v12 = (__int64 *)(a2 + 24);
      v13 = (__int64 *)sub_C33340();
      if ( *((__int64 **)a2 + 3) == v13 )
        sub_C3C790(v27, (_QWORD **)v12);
      else
        sub_C33EB0(v27, v12);
      v26 = 0;
      sub_C41640(v27, v11, 0, &v26);
      if ( v13 == (__int64 *)v27[0] )
        sub_C3C840(&v28, v27);
      else
        sub_C338E0((__int64)&v28, (__int64)v27);
      v14 = sub_AD8F10(v7, (__int64 *)&v28);
      if ( v28 == v13 )
      {
        if ( v29 )
        {
          for ( i = (_QWORD *)(v29 + 24LL * *(_QWORD *)(v29 - 8)); (_QWORD *)v29 != i; sub_91D830(i) )
            i -= 3;
          j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
        }
      }
      else
      {
        sub_C338F0((__int64)&v28);
      }
      sub_91D830(v27);
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 > 1 )
        BUG();
      v28 = (__int64 *)v30;
      v29 = 0x800000000LL;
      v15 = *(_DWORD *)(v4 + 32);
      if ( v15 <= 0 )
      {
        v22 = 0;
        v23 = (__int64 *)v30;
      }
      else
      {
        v16 = 0;
        do
        {
          v17 = sub_AD69F0(a2, v16);
          v18 = sub_2492FB0(a1, v17);
          v20 = (unsigned int)v29;
          v21 = (unsigned int)v29 + 1LL;
          if ( v21 > HIDWORD(v29) )
          {
            v25 = v18;
            sub_C8D5F0((__int64)&v28, v30, (unsigned int)v29 + 1LL, 8u, v19, v21);
            v20 = (unsigned int)v29;
            v18 = v25;
          }
          ++v16;
          v28[v20] = v18;
          v22 = (unsigned int)(v29 + 1);
          LODWORD(v29) = v29 + 1;
        }
        while ( v15 != v16 );
        v23 = v28;
      }
      v14 = sub_AD3730(v23, v22);
      if ( v28 != (__int64 *)v30 )
        _libc_free((unsigned __int64)v28);
    }
    return v14;
  }
  else
  {
    v5 = (__int64 **)sub_2491640(*a1, v4);
    return sub_ACA8A0(v5);
  }
}
