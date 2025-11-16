// Function: sub_1695660
// Address: 0x1695660
//
__int64 __fastcall sub_1695660(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  char *v4; // rbx
  unsigned __int64 v5; // r14
  char v6; // r13
  char *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  char *v14; // r13
  char *v15; // rax
  __int64 v16; // rdx
  int v17; // [rsp+0h] [rbp-50h]
  int v18; // [rsp+4h] [rbp-4Ch]
  unsigned int v19; // [rsp+14h] [rbp-3Ch]

  if ( a3 )
  {
    v11 = sub_1695640(a2);
    if ( v11 )
    {
      v13 = (_BYTE *)sub_161E970(*(_QWORD *)(v11 - 8LL * *(unsigned int *)(v11 + 8)));
      *(_QWORD *)a1 = a1 + 16;
      if ( v13 )
      {
        sub_1693C00((__int64 *)a1, v13, (__int64)&v13[v12]);
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0;
        *(_BYTE *)(a1 + 16) = 0;
      }
    }
    else
    {
      v15 = (char *)sub_1649960(a2);
      sub_16949D0((__int64 *)a1, v15, v16, 0, byte_3F871B3, 0);
    }
  }
  else
  {
    v3 = *(_QWORD *)(a2 + 40);
    v4 = *(char **)(v3 + 176);
    v5 = *(_QWORD *)(v3 + 184);
    if ( byte_4F9FA40 || (v4 = (char *)sub_16C40A0(v4, v5, 2), v5 = v10, byte_4F9FA40) )
    {
      v18 = dword_4F9F960;
      if ( dword_4F9F960 )
      {
        if ( &v4[v5] != v4 )
        {
          v17 = 0;
          v14 = v4;
          do
          {
            v19 = (_DWORD)v14 + 1 - (_DWORD)v4;
            if ( (unsigned __int8)sub_16C36C0((unsigned int)*v14, 2) )
            {
              if ( !--v18 )
                goto LABEL_18;
              v17 = (_DWORD)v14 + 1 - (_DWORD)v4;
            }
            ++v14;
          }
          while ( &v4[v5] != v14 );
          v19 = v17;
LABEL_18:
          if ( v19 > v5 )
          {
            v4 += v5;
            v5 = 0;
          }
          else
          {
            v5 -= v19;
            v4 += v19;
          }
        }
      }
    }
    v6 = *(_BYTE *)(a2 + 32);
    v7 = (char *)sub_1649960(a2);
    sub_16949D0((__int64 *)a1, v7, v8, v6 & 0xF, v4, v5);
  }
  return a1;
}
