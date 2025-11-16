// Function: sub_3512160
// Address: 0x3512160
//
__int64 __fastcall sub_3512160(_QWORD *a1, __int64 *a2)
{
  char *v3; // rax
  __int64 v4; // rdx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *i; // r12
  __int64 v18; // r14
  unsigned int v19; // eax
  __int64 *v20; // [rsp+0h] [rbp-50h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v20 = a2 + 40;
  if ( a2 + 40 != *(__int64 **)(a2[41] + 8) )
  {
    v3 = (char *)sub_2E791E0(a2);
    if ( sub_BC63A0(v3, v4) )
    {
      v6 = (__int64 *)a1[1];
      v7 = *v6;
      v8 = v6[1];
      if ( v7 == v8 )
LABEL_22:
        BUG();
      while ( *(_UNKNOWN **)v7 != &unk_501F1C8 )
      {
        v7 += 16;
        if ( v8 == v7 )
          goto LABEL_22;
      }
      v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
             *(_QWORD *)(v7 + 8),
             &unk_501F1C8);
      v10 = (__int64 *)a1[1];
      a1[25] = v9 + 169;
      v11 = *v10;
      v12 = v10[1];
      if ( v11 == v12 )
LABEL_21:
        BUG();
      while ( *(_UNKNOWN **)v11 != &unk_501EC08 )
      {
        v11 += 16;
        if ( v12 == v11 )
          goto LABEL_21;
      }
      v13 = (__int64 *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
                          *(_QWORD *)(v11 + 8),
                          &unk_501EC08)
                      + 200);
      a1[26] = v13;
      v14 = a2[41];
      if ( (__int64 *)v14 != v20 )
      {
        while ( 1 )
        {
          v15 = sub_2E39EA0(v13, v14);
          v16 = *(__int64 **)(v14 + 112);
          v21[0] = v15;
          for ( i = &v16[*(unsigned int *)(v14 + 120)]; i != v16; ++v16 )
          {
            v18 = *v16;
            if ( !sub_2E322F0(v14, *v16) )
            {
              v19 = sub_2E441D0(a1[25], v14, v18);
              sub_1098D20(v21, v19);
            }
          }
          v14 = *(_QWORD *)(v14 + 8);
          if ( v20 == (__int64 *)v14 )
            break;
          v13 = (__int64 *)a1[26];
        }
      }
    }
  }
  return 0;
}
