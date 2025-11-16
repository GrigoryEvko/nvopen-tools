// Function: sub_35F3D30
// Address: 0x35f3d30
//
void __fastcall sub_35F3D30(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, const char *a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  _WORD *v11; // rdx
  unsigned __int8 *v12; // rsi

  v7 = *(_QWORD *)(a2 + 16) + 16LL * a3;
  if ( a5 )
  {
    if ( !strcmp(a5, "bypass") )
    {
      v9 = a4[4];
      v10 = a4[3] - v9;
      if ( *(_QWORD *)(v7 + 8) )
      {
        if ( v10 > 2 )
        {
          *(_BYTE *)(v9 + 2) = 103;
          *(_WORD *)v9 = 25390;
          a4[4] += 3LL;
          return;
        }
        v12 = (unsigned __int8 *)&unk_435F090;
      }
      else
      {
        if ( v10 > 2 )
        {
          *(_BYTE *)(v9 + 2) = 97;
          *(_WORD *)v9 = 25390;
          a4[4] += 3LL;
          return;
        }
        v12 = (unsigned __int8 *)&unk_435F08C;
      }
      sub_CB6200((__int64)a4, v12, 3u);
      return;
    }
    if ( !strcmp(a5, "srcsize") && (*(_BYTE *)v7 != 2 || *(_QWORD *)(v7 + 8) != -1) )
    {
      v11 = (_WORD *)a4[4];
      if ( a4[3] - (_QWORD)v11 <= 1u )
      {
        sub_CB6200((__int64)a4, (unsigned __int8 *)", ", 2u);
      }
      else
      {
        *v11 = 8236;
        a4[4] += 2LL;
      }
      sub_35EE840(a1, a2, a3, a4, (__int64)a5, a6);
    }
  }
}
