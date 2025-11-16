// Function: sub_2AAAEA0
// Address: 0x2aaaea0
//
__int64 __fastcall sub_2AAAEA0(_QWORD *a1, int *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  int v4; // ecx
  char v5; // r14
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v10; // rdi
  int v11; // [rsp+Ch] [rbp-44h]
  __int64 v12; // [rsp+18h] [rbp-38h]

  v2 = a1[1];
  if ( *(_BYTE *)v2 != 67 )
    return 0;
  v3 = *(_QWORD *)(v2 - 32);
  v4 = *a2;
  v5 = *((_BYTE *)a2 + 4);
  v6 = *(_QWORD *)(v3 + 8);
  v7 = *(_QWORD *)(*a1 + 40LL);
  if ( ((*(_BYTE *)(v6 + 8) - 7) & 0xFD) != 0 )
  {
    if ( !v5 && v4 == 1 )
    {
      v8 = *(_QWORD *)(v2 + 8);
      goto LABEL_12;
    }
    v11 = *a2;
    BYTE4(v12) = *((_BYTE *)a2 + 4);
    LODWORD(v12) = *a2;
    sub_BCE1B0((__int64 *)v6, v12);
    v4 = v11;
  }
  v8 = *(_QWORD *)(v2 + 8);
  LODWORD(v12) = v4;
  BYTE4(v12) = v5;
  if ( ((*(_BYTE *)(v8 + 8) - 7) & 0xFD) != 0 && (v5 || v4 != 1) )
    v8 = sub_BCE1B0((__int64 *)v8, v12);
  v3 = *(_QWORD *)(v2 - 32);
LABEL_12:
  v10 = *(_QWORD *)(v7 + 440);
  if ( v3 != *(_QWORD *)(v10 + 72) )
  {
    if ( !(unsigned __int8)sub_DFA860(*(_QWORD *)(v7 + 448)) )
    {
      v10 = *(_QWORD *)(v7 + 440);
      return sub_31A68A0(v10, v3, v8);
    }
    return 0;
  }
  return sub_31A68A0(v10, v3, v8);
}
