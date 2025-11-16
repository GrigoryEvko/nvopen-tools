// Function: sub_AC5BE0
// Address: 0xac5be0
//
__int64 __fastcall sub_AC5BE0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r14
  __int64 v3; // rdx
  __int64 v4; // r13
  unsigned int v5; // eax
  int v6; // eax
  _QWORD *v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 *v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+8h] [rbp-38h]
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]

  v1 = ***(_QWORD ***)(a1 + 8);
  v2 = sub_AC52D0(a1);
  v4 = v3;
  v5 = sub_C92610(v2, v3);
  v6 = sub_C92860(v1 + 1968, v2, v4, v5);
  if ( v6 == -1 )
  {
    v7 = (_QWORD *)(*(_QWORD *)(v1 + 1968) + 8LL * *(unsigned int *)(v1 + 1976));
    v8 = *v7;
    v9 = *(_QWORD *)(*v7 + 8LL);
    v10 = *(_QWORD *)(v9 + 32);
    if ( v10 )
    {
LABEL_3:
      if ( a1 == v9 )
      {
        v27 = (__int64 *)(v8 + 8);
      }
      else
      {
        while ( a1 != v10 )
        {
          v9 = v10;
          v10 = *(_QWORD *)(v10 + 32);
        }
        v27 = (__int64 *)(v9 + 32);
      }
      result = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = 0;
      v12 = *v27;
      *v27 = result;
      if ( v12 )
      {
        v13 = *(_QWORD *)(v12 + 32);
        if ( v13 )
        {
          v14 = *(_QWORD *)(v13 + 32);
          if ( v14 )
          {
            v15 = *(_QWORD *)(v14 + 32);
            if ( v15 )
            {
              v16 = *(_QWORD *)(v15 + 32);
              if ( v16 )
              {
                v17 = *(_QWORD *)(v16 + 32);
                if ( v17 )
                {
                  v18 = *(_QWORD *)(v17 + 32);
                  if ( v18 )
                  {
                    v28 = *(_QWORD *)(v16 + 32);
                    v30 = *(_QWORD *)(v17 + 32);
                    sub_AC5B80((__int64 *)(v18 + 32));
                    sub_BD7260(v30);
                    sub_BD2DD0(v30);
                    v17 = v28;
                  }
                  v31 = v17;
                  sub_BD7260(v17);
                  sub_BD2DD0(v31);
                }
                sub_BD7260(v16);
                sub_BD2DD0(v16);
              }
              sub_BD7260(v15);
              sub_BD2DD0(v15);
            }
            sub_BD7260(v14);
            sub_BD2DD0(v14);
          }
          sub_BD7260(v13);
          sub_BD2DD0(v13);
        }
        sub_BD7260(v12);
        return sub_BD2DD0(v12);
      }
      return result;
    }
  }
  else
  {
    v7 = (_QWORD *)(*(_QWORD *)(v1 + 1968) + 8LL * v6);
    v8 = *v7;
    v9 = *(_QWORD *)(*v7 + 8LL);
    v10 = *(_QWORD *)(v9 + 32);
    if ( v10 )
      goto LABEL_3;
  }
  v19 = (_QWORD *)sub_BD5C60(a1, v2, v9);
  v20 = (_QWORD *)*v7;
  sub_C929B0(*v19 + 1968LL, *v7);
  v21 = v20[1];
  v22 = *v20 + 17LL;
  if ( v21 )
  {
    v23 = *(_QWORD *)(v21 + 32);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 32);
      if ( v24 )
      {
        v25 = *(_QWORD *)(v24 + 32);
        if ( v25 )
        {
          v26 = *(_QWORD *)(v25 + 32);
          if ( v26 )
          {
            v29 = *(_QWORD *)(v24 + 32);
            v32 = *(_QWORD *)(v25 + 32);
            sub_AC5B80((__int64 *)(v26 + 32));
            sub_BD7260(v32);
            sub_BD2DD0(v32);
            v25 = v29;
          }
          v33 = v25;
          sub_BD7260(v25);
          sub_BD2DD0(v33);
        }
        sub_BD7260(v24);
        sub_BD2DD0(v24);
      }
      sub_BD7260(v23);
      sub_BD2DD0(v23);
    }
    sub_BD7260(v21);
    sub_BD2DD0(v21);
  }
  return sub_C7D6A0(v20, v22, 8);
}
