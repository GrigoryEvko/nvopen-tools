// Function: sub_30F54A0
// Address: 0x30f54a0
//
unsigned __int64 __fastcall sub_30F54A0(__int64 a1, char *a2, __int64 a3)
{
  signed __int64 v6; // r13
  __int64 v7; // rax
  __int64 i; // rcx
  __int64 **v9; // rsi
  __int64 v10; // r12
  __int64 **v11; // r12
  unsigned __int64 v12; // r14
  signed __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rcx
  bool v18; // cc
  __int64 **v19; // [rsp+10h] [rbp-40h]

  if ( !(unsigned __int8)sub_D4B3D0((__int64)a2) )
    return 0;
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 80);
  for ( i = v7 + 16LL * *(unsigned int *)(a1 + 88); v7 != i; v7 += 16 )
  {
    if ( a2 != *(char **)v7 )
    {
      if ( is_mul_ok(*(unsigned int *)(v7 + 8), v6) )
      {
        v6 *= *(unsigned int *)(v7 + 8);
      }
      else if ( !*(_DWORD *)(v7 + 8) || (v18 = v6 <= 0, v6 = 0x7FFFFFFFFFFFFFFFLL, v18) )
      {
        v6 = 0x8000000000000000LL;
      }
    }
  }
  v9 = *(__int64 ***)a3;
  v10 = 10LL * *(unsigned int *)(a3 + 8);
  v19 = &v9[v10];
  if ( v9 != &v9[v10] )
  {
    v11 = v9;
    v12 = 0;
    while ( 1 )
    {
      v13 = sub_30F5470(a1, v11, a2);
      v15 = v6 * v13;
      if ( is_mul_ok(v6, v13) )
      {
        v16 = v15 + v12;
        if ( __OFADD__(v15, v12) )
        {
          v12 = 0x8000000000000000LL;
          if ( v15 > 0 )
            v12 = 0x7FFFFFFFFFFFFFFFLL;
          goto LABEL_13;
        }
      }
      else
      {
        if ( v13 <= 0 )
        {
          if ( v6 >= 0 || v13 >= 0 )
          {
LABEL_23:
            if ( v14 == 1 )
            {
              v17 = v12 + 0x8000000000000000LL;
              if ( __OFADD__(v12, 0x8000000000000000LL) )
              {
                v12 = 0x8000000000000000LL;
                goto LABEL_13;
              }
LABEL_27:
              v12 = v17;
              goto LABEL_13;
            }
            v16 = v12 + 0x8000000000000000LL;
            if ( __OFADD__(v12, 0x8000000000000000LL) )
            {
              v12 = 0x8000000000000000LL;
              goto LABEL_13;
            }
            goto LABEL_12;
          }
        }
        else if ( v6 <= 0 )
        {
          goto LABEL_23;
        }
        if ( v14 == 1 )
        {
          v17 = v12 + 0x7FFFFFFFFFFFFFFFLL;
          if ( __OFADD__(v12, 0x7FFFFFFFFFFFFFFFLL) )
          {
            v12 = 0x7FFFFFFFFFFFFFFFLL;
            goto LABEL_13;
          }
          goto LABEL_27;
        }
        v16 = v12 + 0x7FFFFFFFFFFFFFFFLL;
        if ( __OFADD__(v12, 0x7FFFFFFFFFFFFFFFLL) )
        {
          v12 = 0x7FFFFFFFFFFFFFFFLL;
          goto LABEL_13;
        }
      }
LABEL_12:
      v12 = v16;
LABEL_13:
      v11 += 10;
      if ( v19 == v11 )
        return v12;
    }
  }
  return 0;
}
