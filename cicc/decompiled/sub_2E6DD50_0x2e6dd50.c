// Function: sub_2E6DD50
// Address: 0x2e6dd50
//
__int64 __fastcall sub_2E6DD50(__int64 a1)
{
  __int64 *v1; // rax
  __int64 *v2; // rsi
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r12
  int v6; // ecx
  void *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdi
  void *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax

  v1 = *(__int64 **)(a1 + 24);
  v2 = &v1[*(unsigned int *)(a1 + 32)];
  if ( v2 == v1 )
    return 1;
  do
  {
    v3 = *v1;
    if ( *v1 )
    {
      v4 = *(_QWORD *)v3;
      if ( *(_QWORD *)v3 )
      {
        v5 = *(_QWORD *)(v3 + 8);
        v6 = *(_DWORD *)(v3 + 16);
        if ( v5 )
        {
          if ( *(_DWORD *)(v5 + 16) + 1 != v6 )
          {
            v14 = sub_CB72A0();
            v15 = sub_904010((__int64)v14, "Node ");
            v16 = sub_2E6C890(v15, v4);
            v17 = sub_904010(v16, " has level ");
            v18 = sub_CB59D0(v17, *(unsigned int *)(v3 + 16));
            v19 = sub_904010(v18, " while its IDom ");
            v20 = sub_2E6C890(v19, *(_QWORD *)v5);
            v21 = sub_904010(v20, " has level ");
            v22 = sub_CB59D0(v21, *(unsigned int *)(v5 + 16));
            sub_904010(v22, "!\n");
            v12 = (__int64 *)sub_CB72A0();
            if ( v12[4] != v12[2] )
LABEL_10:
              sub_CB5AE0(v12);
            return 0;
          }
        }
        else if ( v6 )
        {
          v7 = sub_CB72A0();
          v8 = sub_904010((__int64)v7, "Node without an IDom ");
          v9 = sub_2E6C890(v8, v4);
          v10 = sub_904010(v9, " has a nonzero level ");
          v11 = sub_CB59D0(v10, *(unsigned int *)(v3 + 16));
          sub_904010(v11, "!\n");
          v12 = (__int64 *)sub_CB72A0();
          if ( v12[2] != v12[4] )
            goto LABEL_10;
          return 0;
        }
      }
    }
    ++v1;
  }
  while ( v2 != v1 );
  return 1;
}
