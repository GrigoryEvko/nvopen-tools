// Function: sub_133ADF0
// Address: 0x133adf0
//
__int64 __fastcall sub_133ADF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  unsigned int v8; // eax
  __int64 v9; // r13
  int v10; // edx
  unsigned int v11; // eax
  __int64 v12; // r14
  __int64 v13; // rax
  _QWORD *v14; // rsi
  __int64 v15; // rdx

  result = 1;
  if ( !(a7 | a6 | a5 | a4) )
  {
    if ( *(_BYTE *)a1 )
      sub_1311640(a1);
    v8 = dword_505F9BC;
    if ( (unsigned int)(2 * dword_505F9BC) >= dword_4F96990 )
      return 0;
    if ( *(char *)(a1 + 1) > 0 )
    {
      v9 = qword_50579C0[0];
      if ( qword_50579C0[0] )
        goto LABEL_26;
      v9 = sub_1300B80(a1, 0, (__int64)&off_49E8000);
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 144);
      if ( v9 )
      {
        v10 = unk_4C6F238;
        if ( unk_4C6F238 > 2u )
          goto LABEL_10;
LABEL_26:
        sub_1315160(a1, v9, 0, 1u);
        return 0;
      }
      v9 = sub_1302AE0(a1, 0);
      if ( *(_BYTE *)a1 )
      {
        v13 = *(_QWORD *)(a1 + 296);
        v14 = (_QWORD *)(a1 + 256);
        v15 = a1 + 856;
        if ( v13 )
        {
          if ( v9 == v13 )
          {
            v10 = unk_4C6F238;
            if ( unk_4C6F238 <= 2u )
              goto LABEL_26;
            v8 = dword_505F9BC;
LABEL_10:
            if ( v10 == 4 && v8 > 1 )
              v8 = (v8 >> 1) - (((v8 & 1) == 0) - 1);
            if ( *(_DWORD *)(v9 + 78928) < v8 && a1 != *(_QWORD *)(v9 + 16) )
            {
              v11 = sched_getcpu();
              if ( unk_4C6F238 != 3 && dword_505F9BC >> 1 <= v11 )
                v11 -= dword_505F9BC >> 1;
              if ( *(_DWORD *)(v9 + 78928) != v11 )
              {
                v9 = *(_QWORD *)(a1 + 144);
                if ( v11 != *(_DWORD *)(v9 + 78928) )
                {
                  v12 = qword_50579C0[v11];
                  if ( !v12 )
                    v12 = sub_1300B80(a1, v11, (__int64)&off_49E8000);
                  sub_1302A70(a1, v9, v12);
                  if ( *(_BYTE *)a1 )
                    sub_1311F50(a1, (_QWORD *)(a1 + 256), a1 + 856, v12);
                  v9 = *(_QWORD *)(a1 + 144);
                }
              }
              *(_QWORD *)(v9 + 16) = a1;
            }
            goto LABEL_26;
          }
          sub_1311F50(a1, v14, v15, v9);
        }
        else
        {
          sub_13114E0(a1, v14, v15, v9);
        }
      }
      v10 = unk_4C6F238;
      if ( unk_4C6F238 > 2u )
      {
        v8 = dword_505F9BC;
        goto LABEL_10;
      }
    }
    if ( !v9 )
      return 0;
    goto LABEL_26;
  }
  return result;
}
