// Function: sub_2EEFB40
// Address: 0x2eefb40
//
__int64 __fastcall sub_2EEFB40(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v9; // r12
  void *v10; // rdx
  _BYTE *v11; // rax
  __int64 result; // rax
  __int64 v13; // rax

  v5 = a2;
  v9 = *(_QWORD *)(a1 + 16);
  v10 = *(void **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 0xEu )
  {
    v13 = sub_CB6200(v9, "- liverange:   ", 0xFu);
    v5 = a2;
    v9 = v13;
  }
  else
  {
    qmemcpy(v10, "- liverange:   ", 15);
    *(_QWORD *)(v9 + 32) += 15LL;
  }
  sub_2E0B3F0(v5, v9);
  v11 = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v9 + 24) )
  {
    sub_CB5D20(v9, 10);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v11 + 1;
    *v11 = 10;
  }
  sub_2EEFA20(a1, a3);
  result = a5 | a4;
  if ( a5 | a4 )
    return sub_2EEF800(*(_QWORD *)(a1 + 16), a4, a5);
  return result;
}
