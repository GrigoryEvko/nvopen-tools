// Function: sub_297CE50
// Address: 0x297ce50
//
__int64 *__fastcall sub_297CE50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  _BYTE *v6; // r13
  __int64 v7; // r14
  bool v8; // al
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax
  unsigned int v13; // r14d
  __int64 v14; // r13
  int i; // [rsp+14h] [rbp-4Ch]
  __int64 v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h]
  int v19; // [rsp+28h] [rbp-38h]
  __int64 v20; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a4 + 40);
  if ( !v4 )
    goto LABEL_18;
  v6 = *(_BYTE **)(v4 + 8);
  v18 = a4;
  v7 = *(_QWORD *)(a3 + 8);
  i = 0;
  v17 = 0;
  if ( (_BYTE *)v7 == v6 )
    goto LABEL_12;
LABEL_3:
  if ( *(_QWORD *)(a3 + 56) == *(_QWORD *)(v4 + 56) && *(_QWORD *)(a3 + 16) == *(_QWORD *)(v4 + 16) )
  {
    v19 = 2;
    v8 = sub_297BA30(a3, v4, 2);
LABEL_6:
    if ( v8 )
    {
      if ( (unsigned __int8)sub_D9B130(*(_QWORD *)(a2 + 16), (_BYTE *)v7, v6) )
      {
        v9 = sub_DCC810(*(__int64 **)(a2 + 16), v7, (__int64)v6, 0, 0);
        if ( !*((_WORD *)v9 + 12) )
        {
          v17 = v9[4];
          for ( i = v19; ; i = 1 )
          {
            v10 = *(_QWORD *)(v4 + 40);
            v18 = v4;
            if ( !v10 )
              goto LABEL_15;
            v4 = *(_QWORD *)(v4 + 40);
            v7 = *(_QWORD *)(a3 + 8);
            v6 = *(_BYTE **)(v10 + 8);
            if ( (_BYTE *)v7 != v6 )
              goto LABEL_3;
LABEL_12:
            v7 = *(_QWORD *)(a3 + 56);
            v6 = *(_BYTE **)(v4 + 56);
            if ( (_BYTE *)v7 != v6 || *(_QWORD *)(v4 + 32) == *(_QWORD *)(a3 + 32) || *(_DWORD *)v4 != *(_DWORD *)a3 )
              goto LABEL_13;
            v12 = sub_297C710(a3, v4);
            v13 = *(_DWORD *)(v12 + 32);
            v14 = v12;
            if ( v13 <= 0x40 )
            {
              if ( *(_QWORD *)(v12 + 24) <= 1u )
                goto LABEL_29;
            }
            else
            {
              v20 = v12 + 24;
              if ( v13 == (unsigned int)sub_C444A0(v12 + 24) || (unsigned int)sub_C444A0(v20) == v13 - 1 )
                goto LABEL_29;
            }
            v7 = *(_QWORD *)(a3 + 56);
            if ( *(_WORD *)(v7 + 24) )
            {
              v6 = *(_BYTE **)(v4 + 56);
              if ( *(_QWORD *)(a3 + 8) != *(_QWORD *)(v4 + 8) )
              {
                v6 = *(_BYTE **)(v4 + 8);
                v7 = *(_QWORD *)(a3 + 8);
                goto LABEL_3;
              }
LABEL_13:
              if ( *(_QWORD *)(a3 + 16) != *(_QWORD *)(v4 + 16) )
                break;
              v19 = 3;
              v8 = sub_297BA30(a3, v4, 3);
              goto LABEL_6;
            }
LABEL_29:
            v17 = v14;
          }
        }
      }
    }
  }
  v4 = v18;
LABEL_15:
  if ( a4 == v4 )
  {
LABEL_18:
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
  }
  else
  {
    *((_DWORD *)a1 + 2) = i;
    *a1 = v4;
    a1[2] = v17;
  }
  return a1;
}
