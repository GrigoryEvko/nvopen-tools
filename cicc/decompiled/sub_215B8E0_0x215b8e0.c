// Function: sub_215B8E0
// Address: 0x215b8e0
//
__int64 __fastcall sub_215B8E0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r15

  v3 = *a3;
  v4 = *a3 + 16LL * *((unsigned int *)a3 + 2);
  if ( *a3 == v4 )
    return 1;
  v5 = 0;
  while ( 1 )
  {
    if ( **(_WORD **)(*(_QWORD *)v3 + 16LL) != 12 )
      goto LABEL_3;
    v6 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v3 + 32LL) + 8LL);
    v7 = *(_QWORD *)(a2 + 40);
    v8 = (int)v6 < 0
       ? *(_QWORD *)(*(_QWORD *)(v7 + 24) + 16 * (v6 & 0x7FFFFFFF) + 8)
       : *(_QWORD *)(*(_QWORD *)(v7 + 272) + 8 * v6);
    if ( !v8 )
      goto LABEL_3;
    if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 )
      break;
LABEL_13:
    v8 = *(_QWORD *)(v8 + 32);
    if ( v8 )
    {
      if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 )
        break;
      v3 += 16;
      if ( v4 == v3 )
        return 1;
    }
    else
    {
LABEL_3:
      v3 += 16;
      if ( v4 == v3 )
        return 1;
    }
  }
  v9 = *(_QWORD *)(v8 + 16);
  if ( v9 && (unsigned __int8)sub_216EEE0(*(_QWORD *)(v8 + 16)) )
  {
    if ( v5 )
    {
      if ( !(unsigned __int8)sub_1E31610(*(_QWORD *)(v9 + 32) + 240LL, *(_QWORD *)(v5 + 32) + 240LL) )
        return 0;
    }
    else
    {
      v5 = v9;
    }
    goto LABEL_13;
  }
  return 0;
}
