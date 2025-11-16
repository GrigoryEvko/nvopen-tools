// Function: sub_CA5AA0
// Address: 0xca5aa0
//
__int64 __fastcall sub_CA5AA0(__int64 a1, const void *a2, size_t a3, char a4)
{
  __int64 v6; // r12
  __int64 v7; // rdx
  void *v9; // rdi
  __int64 v10; // rdi
  _WORD *v11; // rdx
  _WORD *v12; // r12
  _QWORD v14[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( !a3 )
    goto LABEL_2;
  v9 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v9 < a3 )
  {
    v10 = sub_CB6200(a1, a2, a3);
    v12 = *(_WORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v12 > 1u )
      goto LABEL_7;
  }
  else
  {
    memcpy(v9, a2, a3);
    v10 = a1;
    v11 = (_WORD *)(*(_QWORD *)(a1 + 32) + a3);
    *(_QWORD *)(a1 + 32) = v11;
    v12 = v11;
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v11 > 1u )
    {
LABEL_7:
      *v12 = 8250;
      *(_QWORD *)(v10 + 32) += 2LL;
      goto LABEL_2;
    }
  }
  sub_CB6200(v10, ": ", 2);
LABEL_2:
  sub_CA57B0((__int64)v14, a1, 7, 2 * (a4 != 0));
  v6 = v14[0];
  v7 = *(_QWORD *)(v14[0] + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(v14[0] + 24LL) - v7) <= 8 )
  {
    v6 = sub_CB6200(v14[0], "warning: ", 9);
  }
  else
  {
    *(_BYTE *)(v7 + 8) = 32;
    *(_QWORD *)v7 = 0x3A676E696E726177LL;
    *(_QWORD *)(v6 + 32) += 9LL;
  }
  sub_CA5960((__int64)v14);
  return v6;
}
