// Function: sub_27C0750
// Address: 0x27c0750
//
void __fastcall sub_27C0750(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // rbx
  char v5; // al
  __int64 v6; // r14
  __int64 *v7; // r13
  __int64 v8; // r14
  __int64 *i; // r15
  __int64 v10; // r13
  __int64 v11; // r13
  char v12; // al
  __int64 *v13; // r13
  char v14; // al
  char v15; // al

  if ( a1 != a2 )
  {
    v3 = a1 + 1;
    if ( a2 != a1 + 1 )
    {
      while ( 1 )
      {
        v8 = *v3;
        i = v3;
        v10 = *a1;
        if ( *a1 == *v3 )
          goto LABEL_9;
        sub_B196A0(*(_QWORD *)(a3 + 16), *v3, *a1);
        if ( v5 )
        {
          v6 = *v3;
          v7 = v3 + 1;
          if ( a1 != v3 )
            memmove(a1 + 1, a1, (char *)v3 - (char *)a1);
          ++v3;
          *a1 = v6;
          if ( a2 == v7 )
            return;
        }
        else
        {
          sub_B196A0(*(_QWORD *)(a3 + 16), v10, v8);
          if ( !v15 )
            goto LABEL_21;
          v8 = *v3;
LABEL_9:
          v11 = *(v3 - 1);
          if ( v8 == v11 )
          {
LABEL_14:
            *i = v8;
            v13 = v3 + 1;
          }
          else
          {
            for ( i = v3 - 1; ; --i )
            {
              sub_B196A0(*(_QWORD *)(a3 + 16), v8, v11);
              if ( !v12 )
                break;
              v11 = *(i - 1);
              i[1] = *i;
              if ( v8 == v11 )
                goto LABEL_14;
            }
            sub_B196A0(*(_QWORD *)(a3 + 16), v11, v8);
            if ( !v14 )
LABEL_21:
              BUG();
            v13 = v3 + 1;
            i[1] = v8;
          }
          v3 = v13;
          if ( a2 == v13 )
            return;
        }
      }
    }
  }
}
