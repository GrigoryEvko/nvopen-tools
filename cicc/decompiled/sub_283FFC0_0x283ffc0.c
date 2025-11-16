// Function: sub_283FFC0
// Address: 0x283ffc0
//
__int64 __fastcall sub_283FFC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v3; // r13
  __int64 v4; // r14
  __int64 *v5; // r15
  __int64 *v7; // r14
  unsigned int v8; // r13d
  __int64 *v9; // rax

  v3 = *(_WORD *)(a3 + 2);
  v4 = *(_QWORD *)(a3 - 32);
  v5 = sub_DD8400(*(_QWORD *)(a2 + 16), *(_QWORD *)(a3 - 64));
  if ( sub_D96A50((__int64)v5) )
    goto LABEL_2;
  v7 = sub_DD8400(*(_QWORD *)(a2 + 16), v4);
  if ( sub_D96A50((__int64)v7) )
    goto LABEL_2;
  v8 = v3 & 0x3F;
  if ( sub_DADE90(*(_QWORD *)(a2 + 16), (__int64)v5, *(_QWORD *)(a2 + 40)) )
  {
    v8 = sub_B52F50(v8);
    v9 = v5;
    v5 = v7;
    v7 = v9;
  }
  if ( *((_WORD *)v5 + 12) != 8 || *(_QWORD *)(a2 + 40) != v5[6] )
  {
LABEL_2:
    *(_BYTE *)(a1 + 24) = 0;
  }
  else
  {
    *(_DWORD *)a1 = v8;
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 16) = v7;
    *(_BYTE *)(a1 + 24) = 1;
  }
  return a1;
}
