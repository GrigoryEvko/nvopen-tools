// Function: sub_2EEF900
// Address: 0x2eef900
//
__int64 __fastcall sub_2EEF900(__int64 a1, unsigned int *a2)
{
  void *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  _WORD *v6; // rdx
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 0xEu )
  {
    a1 = sub_CB6200(a1, "- ValNo:       ", 0xFu);
  }
  else
  {
    qmemcpy(v2, "- ValNo:       ", 15);
    *(_QWORD *)(a1 + 32) += 15LL;
  }
  v3 = sub_CB59D0(a1, *a2);
  v4 = *(_QWORD *)(v3 + 32);
  v5 = v3;
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v4) <= 5 )
  {
    v5 = sub_CB6200(v3, " (def ", 6u);
  }
  else
  {
    *(_DWORD *)v4 = 1701062688;
    *(_WORD *)(v4 + 4) = 8294;
    *(_QWORD *)(v3 + 32) += 6LL;
  }
  v8[0] = *((_QWORD *)a2 + 1);
  sub_2FAD600(v8, v5);
  v6 = *(_WORD **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 1u )
    return sub_CB6200(v5, (unsigned __int8 *)")\n", 2u);
  *v6 = 2601;
  *(_QWORD *)(v5 + 32) += 2LL;
  return 2601;
}
