// Function: sub_2765BC0
// Address: 0x2765bc0
//
__int64 __fastcall sub_2765BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 i; // r14
  bool v8; // al
  __int64 v9; // rbx
  __int64 *v10; // r15
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 *v16; // r13
  __int64 v18; // r8
  __int64 *v19; // rax
  __int64 v21; // [rsp+10h] [rbp-50h]

  v6 = (a3 - 1) / 2;
  v21 = a3 & 1;
  if ( a2 >= v6 )
  {
    v10 = (__int64 *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_18;
    v9 = a2;
    goto LABEL_21;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v12 = 32 * (i + 1);
    v13 = (__int64 *)(a1 + v12 - 16);
    v10 = (__int64 *)(a1 + v12);
    v14 = *(_QWORD *)(a1 + v12);
    if ( v14 == *v13 )
      v8 = sub_B445A0(v10[1], v13[1]);
    else
      v8 = sub_B445A0(v14, *v13);
    if ( v8 )
    {
      --v9;
      v10 = (__int64 *)(a1 + 16 * v9);
    }
    v11 = (__int64 *)(a1 + 16 * i);
    *v11 = *v10;
    v11[1] = v10[1];
    if ( v9 >= v6 )
      break;
  }
  if ( !v21 )
  {
LABEL_21:
    if ( (a3 - 2) / 2 == v9 )
    {
      v18 = v9 + 1;
      v9 = 2 * (v9 + 1) - 1;
      v19 = (__int64 *)(a1 + 32 * v18 - 16);
      *v10 = *v19;
      v10[1] = v19[1];
      v10 = (__int64 *)(a1 + 16 * v9);
    }
  }
  v15 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v16 = (__int64 *)(a1 + 16 * v15);
      if ( *v16 == a4 )
      {
        v10 = (__int64 *)(a1 + 16 * v9);
        if ( !sub_B445A0(v16[1], a5) )
          goto LABEL_18;
      }
      else
      {
        v10 = (__int64 *)(a1 + 16 * v9);
        if ( !sub_B445A0(*v16, a4) )
          goto LABEL_18;
      }
      v9 = v15;
      *v10 = *v16;
      v10[1] = v16[1];
      if ( a2 >= v15 )
        break;
      v15 = (v15 - 1) / 2;
    }
    v10 = (__int64 *)(a1 + 16 * v15);
  }
LABEL_18:
  *v10 = a4;
  v10[1] = a5;
  return a5;
}
