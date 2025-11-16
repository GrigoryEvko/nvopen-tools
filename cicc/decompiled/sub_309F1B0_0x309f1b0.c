// Function: sub_309F1B0
// Address: 0x309f1b0
//
void __fastcall sub_309F1B0(int a1, char a2, __int64 a3, __int64 a4)
{
  _QWORD *v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  _WORD *v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rdx
  _BYTE *v14; // rax

  if ( (_BYTE)qword_502E088 || a2 )
  {
    v7 = sub_CB72A0();
    v8 = (_WORD *)v7[4];
    v9 = (__int64)v7;
    if ( v7[3] - (_QWORD)v8 <= 1u )
    {
      v9 = sub_CB6200((__int64)v7, (unsigned __int8 *)"  ", 2u);
    }
    else
    {
      *v8 = 8224;
      v7[4] += 2LL;
    }
    v10 = sub_CF5E90(v9, a1);
    v11 = *(_WORD **)(v10 + 32);
    v12 = v10;
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 1u )
    {
      v12 = sub_CB6200(v10, (unsigned __int8 *)": ", 2u);
    }
    else
    {
      *v11 = 8250;
      *(_QWORD *)(v10 + 32) += 2LL;
    }
    sub_A69870(a3, (_BYTE *)v12, 0);
    v13 = *(_QWORD *)(v12 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v12 + 24) - v13) <= 4 )
    {
      v12 = sub_CB6200(v12, " <-> ", 5u);
    }
    else
    {
      *(_DWORD *)v13 = 1043151904;
      *(_BYTE *)(v13 + 4) = 32;
      *(_QWORD *)(v12 + 32) += 5LL;
    }
    sub_A69870(a4, (_BYTE *)v12, 0);
    v14 = *(_BYTE **)(v12 + 32);
    if ( (unsigned __int64)v14 >= *(_QWORD *)(v12 + 24) )
    {
      sub_CB5D20(v12, 10);
    }
    else
    {
      *(_QWORD *)(v12 + 32) = v14 + 1;
      *v14 = 10;
    }
  }
}
