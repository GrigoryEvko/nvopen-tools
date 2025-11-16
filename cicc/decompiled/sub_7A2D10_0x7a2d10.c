// Function: sub_7A2D10
// Address: 0x7a2d10
//
__int64 __fastcall sub_7A2D10(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v4; // rbx
  char v5; // al
  __int64 v6; // r14
  __int64 i; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 result; // rax

  v4 = (__int64 *)*a4;
  v5 = *(_BYTE *)(*a4 + 8);
  if ( (v5 & 1) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      return 0;
    sub_6855B0(0xA8Du, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    return 0;
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 240);
    if ( (v5 & 4) != 0 && !(unsigned int)sub_777F30(a1, *a4, (FILE *)(a3 + 28)) )
      return 0;
    qword_4F08068 = *v4;
    for ( i = *(_QWORD *)(v6 + 32); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    qword_4F08060 = i;
    v8 = sub_72B840(a2);
    v12 = *(_QWORD *)(v8 + 80);
    if ( *(_BYTE *)(v12 + 40) == 19 )
      v12 = *(_QWORD *)(*(_QWORD *)(v12 + 72) + 8LL);
    result = sub_7987E0(a1, *(_QWORD *)(v12 + 72), v8, v9, v10, v11);
    qword_4F08060 = 0;
  }
  return result;
}
