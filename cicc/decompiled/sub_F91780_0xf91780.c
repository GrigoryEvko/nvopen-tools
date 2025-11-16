// Function: sub_F91780
// Address: 0xf91780
//
bool __fastcall sub_F91780(__int64 **a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  __int64 v3; // r14
  unsigned int i; // r15d
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 j; // rbx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  int v13; // [rsp+14h] [rbp-3Ch]
  __int64 v14; // [rsp+18h] [rbp-38h]

  v1 = **a1;
  v2 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 != v1 + 48 )
  {
    if ( !v2 )
      BUG();
    v3 = v2 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 <= 0xA )
    {
      v13 = sub_B46E30(v3);
      if ( v13 )
      {
        for ( i = 0; i != v13; ++i )
        {
          v5 = sub_B46EC0(v3, i);
          v6 = *(_QWORD *)(v5 + 56);
          for ( j = v5 + 48; j != v6; v6 = *(_QWORD *)(v6 + 8) )
          {
            if ( !v6 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
            {
              if ( !sub_F8F4A0((unsigned __int8 *)(v6 - 24), *a1[1]) )
                return 0;
              v10 = (__int64)a1[2];
              v11 = *(unsigned int *)(v10 + 8);
              if ( (_DWORD)qword_4F8CFC8 == (_DWORD)v11 )
                return 0;
              v12 = v6 - 24;
              if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 12) )
              {
                v14 = (__int64)a1[2];
                sub_C8D5F0(v14, (const void *)(v10 + 16), v11 + 1, 8u, v9, v12);
                v10 = v14;
                v12 = v6 - 24;
                v11 = *(unsigned int *)(v14 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v10 + 8 * v11) = v12;
              ++*(_DWORD *)(v10 + 8);
            }
            else if ( (unsigned int)sub_B46E30(v6 - 24) > 1 )
            {
              return 0;
            }
          }
        }
      }
    }
  }
  return *((_DWORD *)a1[2] + 2) != 0;
}
