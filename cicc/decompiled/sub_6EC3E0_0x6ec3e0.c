// Function: sub_6EC3E0
// Address: 0x6ec3e0
//
__int64 __fastcall sub_6EC3E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v9; // r14
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // [rsp+4h] [rbp-3Ch]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v5 = a1;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_QWORD *)(v6 + 32);
  if ( v7 != a3 )
  {
    v9 = a2;
    if ( !(unsigned int)sub_8D97D0(a3, *(_QWORD *)(v6 + 32), 0, a4, a5) )
    {
      v11 = sub_8D5CE0(a3, v7);
      if ( (*(_BYTE *)(v11 + 96) & 4) != 0 )
      {
        v19 = sub_8D2E30(*(_QWORD *)a2);
        if ( *(_BYTE *)(a2 + 24) == 1 )
        {
          v20 = v7;
          do
          {
            if ( v7 )
            {
              if ( a3 )
              {
                if ( dword_4F07588 )
                {
                  v15 = *(_QWORD *)(v7 + 32);
                  if ( *(_QWORD *)(a3 + 32) == v15 )
                  {
                    if ( v15 )
                      break;
                  }
                }
              }
            }
            if ( *(_BYTE *)(v9 + 56) == 14 )
            {
              v7 = **(_QWORD **)(v9 + 72);
              if ( v19 )
                v7 = sub_8D46C0(**(_QWORD **)(v9 + 72));
              while ( *(_BYTE *)(v7 + 140) == 12 )
                v7 = *(_QWORD *)(v7 + 160);
              v16 = *(_QWORD **)(sub_8D5CE0(v7, v20) + 120);
              if ( v16 )
              {
                while ( 1 )
                {
                  v18 = v16[2];
                  if ( v18 == v5 )
                    break;
                  if ( v5 )
                  {
                    if ( v18 )
                    {
                      if ( dword_4F07588 )
                      {
                        v17 = *(_QWORD *)(v18 + 32);
                        if ( *(_QWORD *)(v5 + 32) == v17 )
                        {
                          if ( v17 )
                            break;
                        }
                      }
                    }
                  }
                  v16 = (_QWORD *)*v16;
                  if ( !v16 )
                    goto LABEL_16;
                }
                v20 = v7;
                v5 = v16[1];
              }
            }
LABEL_16:
            v9 = *(_QWORD *)(v9 + 72);
            if ( *(_BYTE *)(v9 + 24) != 1 )
              break;
          }
          while ( v7 != a3 );
        }
      }
      else
      {
        v12 = *(_QWORD **)(v11 + 120);
        if ( v12 )
        {
          while ( 1 )
          {
            v14 = v12[2];
            if ( v14 == a1 )
              break;
            if ( v14 )
            {
              if ( dword_4F07588 )
              {
                v13 = *(_QWORD *)(v14 + 32);
                if ( *(_QWORD *)(a1 + 32) == v13 )
                {
                  if ( v13 )
                    break;
                }
              }
            }
            v12 = (_QWORD *)*v12;
            if ( !v12 )
              return v5;
          }
          return v12[1];
        }
      }
    }
  }
  return v5;
}
