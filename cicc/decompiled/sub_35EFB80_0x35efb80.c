// Function: sub_35EFB80
// Address: 0x35efb80
//
void __fastcall sub_35EFB80(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  _BYTE *v12; // rax
  _QWORD *v13; // rcx
  unsigned int v14; // edx
  _WORD *v15; // rdx

  sub_35EE840(a1, a2, a3, a4, a5, a6);
  if ( a5 && strlen((const char *)a5) == 3 && *(_WORD *)a5 == 25697 && *(_BYTE *)(a5 + 2) == 100 )
  {
    v15 = (_WORD *)a4[4];
    if ( a4[3] - (_QWORD)v15 <= 1u )
    {
      sub_CB6200((__int64)a4, (unsigned __int8 *)", ", 2u);
    }
    else
    {
      *v15 = 8236;
      a4[4] += 2LL;
    }
    v14 = a3 + 1;
    v13 = a4;
LABEL_9:
    sub_35EE840(a1, a2, v14, v13, v9, v10);
    return;
  }
  v11 = *(_QWORD *)(a2 + 16) + 16LL * (a3 + 1);
  if ( *(_BYTE *)v11 != 2 || *(_QWORD *)(v11 + 8) )
  {
    v12 = (_BYTE *)a4[4];
    if ( (_BYTE *)a4[3] == v12 )
    {
      sub_CB6200((__int64)a4, (unsigned __int8 *)"+", 1u);
    }
    else
    {
      *v12 = 43;
      ++a4[4];
    }
    v13 = a4;
    v14 = a3 + 1;
    goto LABEL_9;
  }
}
