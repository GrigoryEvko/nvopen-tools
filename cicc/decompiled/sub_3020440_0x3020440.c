// Function: sub_3020440
// Address: 0x3020440
//
__int64 __fastcall sub_3020440(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v10; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)a2;
  v10 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v10 )
    return 1;
  v3 = 0;
  while ( 1 )
  {
    v4 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (unsigned __int16)(*(_WORD *)(v4 + 68) - 14) > 1u )
      goto LABEL_14;
    v5 = *(unsigned int *)(*(_QWORD *)(v4 + 32) + 8LL);
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 32LL);
    v7 = (int)v5 < 0
       ? *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16 * (v5 & 0x7FFFFFFF) + 8)
       : *(_QWORD *)(*(_QWORD *)(v6 + 304) + 8 * v5);
    if ( !v7 )
      goto LABEL_14;
    if ( (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
      break;
LABEL_12:
    v7 = *(_QWORD *)(v7 + 32);
    if ( v7 && (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
      break;
LABEL_14:
    v2 += 2;
    if ( (_QWORD *)v10 == v2 )
      return 1;
  }
  v8 = *(_QWORD *)(v7 + 16);
  if ( v8 && (unsigned __int8)sub_307AA70(*(_QWORD *)(v7 + 16)) )
  {
    if ( v3 )
    {
      if ( !(unsigned __int8)sub_2EAB6C0(*(_QWORD *)(v8 + 32) + 240LL, (char *)(*(_QWORD *)(v3 + 32) + 240LL)) )
        return 0;
    }
    else
    {
      v3 = v8;
    }
    goto LABEL_12;
  }
  return 0;
}
