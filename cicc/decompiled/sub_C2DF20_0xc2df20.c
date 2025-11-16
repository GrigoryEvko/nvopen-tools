// Function: sub_C2DF20
// Address: 0xc2df20
//
__int64 __fastcall sub_C2DF20(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  __int64 v5; // rax
  _WORD *v6; // rdx
  __int64 v7; // rdi

  v3 = sub_CB6200(a2, a1[1], a1[2]);
  v4 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 24) )
  {
    v3 = sub_CB5D20(v3, 58);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v4 + 1;
    *v4 = 58;
  }
  v5 = sub_CB59F0(v3, a1[5]);
  v6 = *(_WORD **)(v5 + 32);
  v7 = v5;
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 1u )
  {
    v7 = sub_CB6200(v5, ": ", 2);
  }
  else
  {
    *v6 = 8250;
    *(_QWORD *)(v5 + 32) += 2LL;
  }
  return sub_CB6200(v7, a1[6], a1[7]);
}
