// Function: sub_27309A0
// Address: 0x27309a0
//
__int64 __fastcall sub_27309A0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // rax
  __int64 *v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // r12d
  signed __int64 v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  bool v12; // of
  __int64 *v13; // r13
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rsi
  __int64 *v17; // r8
  unsigned int v18; // edx
  bool v20; // cc
  __int64 v22; // [rsp+10h] [rbp-C0h]
  unsigned int v23; // [rsp+18h] [rbp-B8h]
  int v24; // [rsp+1Ch] [rbp-B4h]
  __int64 v26; // [rsp+30h] [rbp-A0h]
  __int64 v27; // [rsp+38h] [rbp-98h]
  __int64 *v28; // [rsp+40h] [rbp-90h]
  unsigned __int64 v30; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+78h] [rbp-58h]
  unsigned __int64 v32; // [rsp+80h] [rbp-50h]
  unsigned int v33; // [rsp+88h] [rbp-48h]
  unsigned __int64 v34; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+98h] [rbp-38h]

  v4 = a2;
  if ( *(_BYTE *)(a1 + 56) && (char *)a3 - (char *)a2 <= 16800 )
  {
    v5 = a2;
    if ( a3 != a2 )
    {
      v22 = -1;
      v24 = 0;
      v23 = 0;
      do
      {
        v6 = v5[18];
        v31 = *(_DWORD *)(v6 + 32);
        if ( v31 > 0x40 )
          sub_C43780((__int64)&v30, (const void **)(v6 + 24));
        else
          v30 = *(_QWORD *)(v6 + 24);
        v7 = *((unsigned int *)v5 + 2);
        v23 += v7;
        v26 = *v5 + 16 * v7;
        if ( *v5 == v26 )
        {
          v8 = 0;
          v9 = 0;
        }
        else
        {
          v27 = *v5;
          v8 = 0;
          v9 = 0;
          do
          {
            v10 = sub_DFB040(*(__int64 **)a1);
            if ( v11 == 1 )
              v8 = 1;
            v12 = __OFADD__(v10, v9);
            v9 += v10;
            if ( v12 )
            {
              v9 = 0x8000000000000000LL;
              if ( v10 > 0 )
                v9 = 0x7FFFFFFFFFFFFFFFLL;
            }
            v13 = a2;
            do
            {
              v16 = v13[18];
              v17 = (__int64 *)(v5[18] + 24);
              v35 = *(_DWORD *)(v16 + 32);
              if ( v35 <= 0x40 )
              {
                v34 = *(_QWORD *)(v16 + 24);
              }
              else
              {
                v28 = v17;
                sub_C43780((__int64)&v34, (const void **)(v16 + 24));
                v17 = v28;
              }
              sub_C46B40((__int64)&v34, v17);
              v33 = v35;
              v32 = v34;
              v14 = sub_DFAFA0(*(_QWORD *)a1);
              if ( v15 == 1 )
                v8 = 1;
              v12 = __OFSUB__(v9, v14);
              v9 -= v14;
              if ( v12 )
              {
                v9 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v14 > 0 )
                  v9 = 0x8000000000000000LL;
              }
              if ( v33 > 0x40 && v32 )
                j_j___libc_free_0_0(v32);
              v13 += 21;
            }
            while ( a3 != v13 );
            v27 += 16;
          }
          while ( v26 != v27 );
        }
        v20 = v24 < v8;
        if ( v24 == v8 )
          v20 = v22 < v9;
        if ( v20 )
        {
          v24 = v8;
          v22 = v9;
          *(_QWORD *)a4 = v5;
        }
        if ( v31 > 0x40 && v30 )
          j_j___libc_free_0_0(v30);
        v5 += 21;
      }
      while ( a3 != v5 );
      return v23;
    }
    return 0;
  }
  if ( a3 == a2 )
    return 0;
  v18 = 0;
  do
  {
    v18 += *((_DWORD *)v4 + 2);
    if ( *((_DWORD *)v4 + 40) > *(_DWORD *)(*(_QWORD *)a4 + 160LL) )
      *(_QWORD *)a4 = v4;
    v4 += 21;
  }
  while ( a3 != v4 );
  return v18;
}
