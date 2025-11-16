// Function: sub_1B42940
// Address: 0x1b42940
//
void __fastcall sub_1B42940(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v3; // rax
  __int64 v5; // rcx
  unsigned int *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int *v10; // rdx
  __int64 v11; // rdx
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a2;
  v5 = 4 * a3;
  v7 = &a2[a3];
  v8 = (4 * a3) >> 4;
  v9 = v5 >> 2;
  if ( v8 > 0 )
  {
    v10 = &a2[4 * v8];
    while ( !*v3 )
    {
      if ( v3[1] )
      {
        ++v3;
        goto LABEL_8;
      }
      if ( v3[2] )
      {
        v3 += 2;
        goto LABEL_8;
      }
      if ( v3[3] )
      {
        v3 += 3;
        goto LABEL_8;
      }
      v3 += 4;
      if ( v3 == v10 )
      {
        v9 = v7 - v3;
        goto LABEL_12;
      }
    }
    goto LABEL_8;
  }
LABEL_12:
  if ( v9 == 2 )
  {
LABEL_19:
    if ( !*v3 )
    {
      ++v3;
LABEL_15:
      v11 = 0;
      if ( !*v3 )
        goto LABEL_10;
      goto LABEL_8;
    }
    goto LABEL_8;
  }
  if ( v9 != 3 )
  {
    v11 = 0;
    if ( v9 != 1 )
      goto LABEL_10;
    goto LABEL_15;
  }
  if ( !*v3 )
  {
    ++v3;
    goto LABEL_19;
  }
LABEL_8:
  v11 = 0;
  if ( v7 != v3 )
  {
    v12[0] = sub_157E9C0(*(_QWORD *)(a1 + 40));
    v11 = sub_161BD30(v12, a2, a3);
  }
LABEL_10:
  sub_1625C10(a1, 2, v11);
}
