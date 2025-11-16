// Function: sub_1291990
// Address: 0x1291990
//
_QWORD *__fastcall sub_1291990(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *result; // rax
  __int64 *v5; // rbx
  __int64 *v6; // r15
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-60h]
  _QWORD *v12; // [rsp+8h] [rbp-58h]
  _QWORD *v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 128LL);
  if ( !v2 )
    sub_127B550("label for goto statement not found!", (_DWORD *)a2, 1);
  sub_1290930(a1, (unsigned int *)a2);
  sub_127C770((_QWORD *)a2);
  if ( *(_BYTE *)(a1 + 168) )
  {
    sub_1291800(&v13, a1, a2);
    v5 = v14;
    v12 = v13;
    v6 = v13;
    v11 = v15;
    while ( v5 != v6 )
    {
      v7 = *v6++;
      sub_127A040(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(v7 + 120));
      sub_12A5710(a1, v7, 1);
    }
    v8 = *(_QWORD *)(a1 + 352);
    v9 = *(_QWORD *)(v8 - 24);
    if ( v9 != *(_QWORD *)(v8 - 16) )
      *(_QWORD *)(v8 - 16) = v9;
    v10 = sub_12A5B50(a1, v2);
    sub_12909B0((_QWORD *)a1, v10);
    result = v12;
    if ( v12 )
      return (_QWORD *)j_j___libc_free_0(v12, v11 - (_QWORD)v12);
  }
  else
  {
    v3 = sub_12A5B50(a1, v2);
    return sub_12909B0((_QWORD *)a1, v3);
  }
  return result;
}
