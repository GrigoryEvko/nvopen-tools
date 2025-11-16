// Function: sub_3881180
// Address: 0x3881180
//
__int64 __fastcall sub_3881180(unsigned __int8 **a1)
{
  unsigned __int8 *v1; // rax
  unsigned __int8 *v2; // rsi
  int v3; // ecx
  unsigned __int8 *v4; // rax
  int v5; // edx
  char *v6; // r12
  unsigned int v7; // r8d
  unsigned __int8 *v9; // rax
  int i; // ecx
  unsigned __int8 *v11; // rcx
  _QWORD *v12; // r14
  __int64 v13; // r13
  void *v14; // rax
  unsigned __int8 *v15; // rax
  unsigned __int8 *v16; // r12
  int v17; // ecx
  unsigned __int8 *v18; // rax
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int8 *v22; // rax
  unsigned __int8 *v23; // r15
  unsigned __int8 *v24; // r13
  unsigned __int8 *j; // rbx
  unsigned __int8 *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // [rsp-68h] [rbp-68h]
  __int64 v30; // [rsp-60h] [rbp-60h]
  __int64 v31; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int8 *v32; // [rsp-50h] [rbp-50h] BYREF
  __int64 v33; // [rsp-48h] [rbp-48h]

  v1 = *a1;
  if ( (unsigned int)**a1 - 48 <= 9 )
  {
    v2 = v1 + 1;
    *a1 = v1 + 1;
    v3 = v1[1];
    if ( (unsigned int)(v3 - 48) <= 9 )
    {
      v4 = v1 + 2;
      do
      {
        *a1 = v4;
        v5 = *v4;
        v2 = v4++;
        LOBYTE(v3) = v5;
      }
      while ( (unsigned int)(v5 - 48) <= 9 );
    }
    v6 = (char *)a1[6];
    if ( (_BYTE)v3 != 46 )
    {
      v7 = 1;
      *a1 = (unsigned __int8 *)(v6 + 1);
      return v7;
    }
    v9 = v2 + 1;
    *a1 = v2 + 1;
    for ( i = v2[1]; (unsigned int)(i - 48) <= 9; i = *v9 )
      *a1 = ++v9;
    if ( (i & 0xDF) != 0x45
      || (v17 = v9[1], (unsigned int)(v17 - 48) > 9)
      && ((((_BYTE)v17 - 43) & 0xFD) != 0 || (unsigned int)v9[2] - 48 > 9) )
    {
      v11 = *a1;
    }
    else
    {
      v11 = v9 + 2;
      *a1 = v9 + 2;
      if ( (unsigned int)v9[2] - 48 <= 9 )
      {
        v18 = v9 + 3;
        do
        {
          v11 = v18;
          *a1 = v18++;
        }
        while ( (unsigned int)*v11 - 48 <= 9 );
      }
    }
    v12 = a1 + 15;
    v13 = v11 - (unsigned __int8 *)v6;
    v14 = sub_1698280();
    sub_169E660((__int64)&v31, v14, v6, v13);
    v15 = (unsigned __int8 *)sub_16982C0();
    v16 = v15;
    if ( a1[15] == v15 )
    {
      v23 = v32;
      v24 = a1[16];
      if ( v15 == v32 )
      {
        if ( v24 )
        {
          v26 = &v24[32 * *((_QWORD *)v24 - 1)];
          while ( v26 != v24 )
          {
            v26 -= 32;
            if ( v23 == *((unsigned __int8 **)v26 + 1) )
            {
              v27 = *((_QWORD *)v26 + 2);
              if ( v27 )
              {
                v28 = v27 + 32LL * *(_QWORD *)(v27 - 8);
                if ( v27 != v28 )
                {
                  do
                  {
                    v29 = v27;
                    v30 = v28 - 32;
                    sub_127D120((_QWORD *)(v28 - 24));
                    v28 = v30;
                    v27 = v29;
                  }
                  while ( v29 != v30 );
                }
                j_j_j___libc_free_0_0(v27 - 8);
              }
            }
            else
            {
              sub_1698460((__int64)(v26 + 8));
            }
          }
          j_j_j___libc_free_0_0((unsigned __int64)(v24 - 8));
        }
        goto LABEL_39;
      }
      if ( !v24 )
        goto LABEL_32;
      for ( j = &v24[32 * *((_QWORD *)v24 - 1)]; v24 != j; sub_127D120((_QWORD *)j + 1) )
        j -= 32;
      j_j_j___libc_free_0_0((unsigned __int64)(v24 - 8));
      v22 = v32;
    }
    else
    {
      if ( v15 != v32 )
      {
        sub_16983E0((__int64)(a1 + 15), (__int64)&v32);
LABEL_15:
        if ( v32 == v16 )
        {
          v19 = v33;
          if ( v33 )
          {
            v20 = 32LL * *(_QWORD *)(v33 - 8);
            v21 = v33 + v20;
            if ( v33 != v33 + v20 )
            {
              do
              {
                v21 -= 32;
                sub_127D120((_QWORD *)(v21 + 8));
              }
              while ( v19 != v21 );
            }
            j_j_j___libc_free_0_0(v19 - 8);
          }
        }
        else
        {
          sub_1698460((__int64)&v32);
        }
        return 389;
      }
      sub_1698460((__int64)(a1 + 15));
      v22 = v32;
    }
    if ( v16 != v22 )
    {
LABEL_32:
      sub_1698450((__int64)v12, (__int64)&v32);
      goto LABEL_15;
    }
LABEL_39:
    sub_169C7E0(v12, &v32);
    goto LABEL_15;
  }
  return 1;
}
