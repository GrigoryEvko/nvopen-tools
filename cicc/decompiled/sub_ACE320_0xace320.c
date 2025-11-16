// Function: sub_ACE320
// Address: 0xace320
//
__int64 __fastcall sub_ACE320(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r13
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  _QWORD *v17; // rdi
  _QWORD v18[4]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v19; // [rsp+20h] [rbp-50h]
  _QWORD v20[9]; // [rsp+28h] [rbp-48h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = 40LL * v6;
  *(_QWORD *)(a1 + 8) = sub_C7D670(v7, 8);
  if ( v5 )
  {
    sub_ACDE10(a1, v5, (int *)(v5 + 40 * v4));
    return sub_C7D6A0(v5, 40 * v4, 8);
  }
  *(_QWORD *)(a1 + 16) = 0;
  v9 = sub_C33690();
  v13 = sub_C33340(v7, 8, v10, v11, v12);
  v14 = v13;
  if ( v9 == v13 )
    sub_C3C5A0(v18, v13, 1);
  else
    sub_C36740(v18, v9, 1);
  LODWORD(v19) = -1;
  BYTE4(v19) = 1;
  if ( v18[0] == v14 )
    sub_C3C840(v20, v18);
  else
    sub_C338E0(v20, v18);
  sub_91D830(v18);
  v15 = *(_QWORD **)(a1 + 8);
  v16 = &v15[5 * *(unsigned int *)(a1 + 24)];
  if ( v15 != v16 )
  {
    while ( 1 )
    {
      if ( !v15 )
        goto LABEL_12;
      v17 = v15 + 1;
      *v15 = v19;
      if ( v14 == v20[0] )
      {
        sub_C3C790(v17, v20);
        v15 += 5;
        if ( v16 == v15 )
          return sub_91D830(v20);
      }
      else
      {
        sub_C33EB0(v17, v20);
LABEL_12:
        v15 += 5;
        if ( v16 == v15 )
          return sub_91D830(v20);
      }
    }
  }
  return sub_91D830(v20);
}
