// Function: sub_26A4990
// Address: 0x26a4990
//
__int64 __fastcall sub_26A4990(_QWORD *a1, int a2)
{
  __int64 v2; // rax
  __int64 *v3; // r9
  __int64 result; // rax
  __int64 *v5; // r15
  __int64 **v6; // rax
  __int64 *v7; // r8
  __int64 *v8; // rbx
  __int64 *i; // r13
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int64 v19; // [rsp+18h] [rbp-88h]
  __int64 *v20; // [rsp+28h] [rbp-78h]
  unsigned __int64 v21; // [rsp+30h] [rbp-70h]

  v19 = a1[9] + 160LL * a2 + 3512;
  v2 = a1[5];
  v3 = *(__int64 **)v2;
  result = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
  v20 = (__int64 *)result;
  if ( v3 != (__int64 *)result )
  {
    v5 = v3;
    do
    {
      v6 = (__int64 **)sub_267FA80(v19, *v5);
      v7 = *v6;
      result = *((unsigned int *)v6 + 2);
      v8 = &v7[result];
      for ( i = v7; v8 != i; result = sub_26A45D0(v17, v21, 0) )
      {
        while ( 1 )
        {
          result = *i;
          v10 = *(_QWORD *)(*i + 24);
          if ( *(_BYTE *)v10 == 85 && result == v10 - 32 )
          {
            if ( *(char *)(v10 + 7) >= 0 )
              goto LABEL_21;
            v18 = *(_QWORD *)(*i + 24);
            v11 = sub_BD2BC0(v18);
            v10 = v18;
            v13 = v11 + v12;
            v14 = 0;
            if ( *(char *)(v18 + 7) < 0 )
            {
              v15 = sub_BD2BC0(v18);
              v10 = v18;
              v14 = v15;
            }
            result = (v13 - v14) >> 4;
            if ( !(_DWORD)result )
            {
LABEL_21:
              result = *(_QWORD *)(v19 + 120);
              if ( result )
              {
                v16 = *(_QWORD *)(v10 - 32);
                if ( v16 )
                {
                  if ( !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *(_QWORD *)(v10 + 80) && result == v16 )
                    break;
                }
              }
            }
          }
          if ( v8 == ++i )
            goto LABEL_18;
        }
        ++i;
        v17 = a1[10];
        v21 = v10 & 0xFFFFFFFFFFFFFFFCLL | 1;
        nullsub_1518();
      }
LABEL_18:
      ++v5;
    }
    while ( v20 != v5 );
  }
  return result;
}
