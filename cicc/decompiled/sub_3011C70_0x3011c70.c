// Function: sub_3011C70
// Address: 0x3011c70
//
void __fastcall sub_3011C70(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rcx
  __int64 *v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 80);
  if ( v2 != a1 + 72 )
  {
    v3 = 0x100060000000001LL;
    do
    {
      v4 = v2 - 24;
      if ( !v2 )
        v4 = 0;
      v5 = sub_AA4FF0(v4);
      if ( !v5 )
        BUG();
      v6 = (unsigned int)*(unsigned __int8 *)(v5 - 24) - 39;
      if ( (unsigned int)v6 <= 0x38 && _bittest64(&v3, v6) )
      {
        v7 = sub_AA4FF0(v4);
        if ( !v7 )
          BUG();
        if ( *(_BYTE *)(v7 - 24) == 81 )
        {
          v8 = *(_QWORD *)(v7 - 56);
          if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
          {
            v9 = *(_QWORD *)(v8 - 8);
            if ( *(_QWORD *)(v9 + 32) )
            {
              v14 = *(_QWORD *)(v9 + 32);
              v10 = sub_AA4FF0(v14);
              if ( !v10 )
                BUG();
              if ( *(_BYTE *)(v10 - 24) == 39 )
              {
                v11 = *(_QWORD *)(v10 - 32);
                v12 = (__int64 *)(v11 + 32);
                v13 = (__int64 *)(v11 + 64);
                if ( (*(_BYTE *)(v10 - 22) & 1) == 0 )
                  v13 = v12;
                sub_3011770(a2, v4, *v13);
              }
              else
              {
                sub_3011770(a2, v4, v14);
              }
            }
          }
        }
      }
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( a1 + 72 != v2 );
  }
}
