// Function: sub_C7EF90
// Address: 0xc7ef90
//
__int64 __fastcall sub_C7EF90(__int64 a1, __int16 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rdx
  _BYTE *v5; // rax
  _WORD *v7; // rdx
  __int64 v8; // rax

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v3) <= 8 )
  {
    sub_CB6200(a1, "captures(", 9);
  }
  else
  {
    *(_BYTE *)(v3 + 8) = 40;
    *(_QWORD *)v3 = 0x7365727574706163LL;
    *(_QWORD *)(a1 + 32) += 9LL;
  }
  if ( HIBYTE(a2) && !(_BYTE)a2 )
  {
    v4 = *(_QWORD *)(a1 + 32);
    goto LABEL_6;
  }
  sub_C7ECD0(a1, a2);
  if ( (_BYTE)a2 == HIBYTE(a2) )
    goto LABEL_8;
  v7 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v7 > 1u )
  {
    *v7 = 8236;
    v4 = *(_QWORD *)(a1 + 32) + 2LL;
    *(_QWORD *)(a1 + 32) = v4;
LABEL_6:
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v4) > 4 )
    {
LABEL_7:
      *(_DWORD *)v4 = 980706674;
      *(_BYTE *)(v4 + 4) = 32;
      *(_QWORD *)(a1 + 32) += 5LL;
      sub_C7ECD0(a1, SHIBYTE(a2));
      goto LABEL_8;
    }
    goto LABEL_14;
  }
  a1 = sub_CB6200(a1, ", ", 2);
  v4 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v4) > 4 )
    goto LABEL_7;
LABEL_14:
  v8 = sub_CB6200(a1, "ret: ", 5);
  sub_C7ECD0(v8, SHIBYTE(a2));
LABEL_8:
  v5 = *(_BYTE **)(v2 + 32);
  if ( *(_BYTE **)(v2 + 24) == v5 )
  {
    sub_CB6200(v2, ")", 1);
  }
  else
  {
    *v5 = 41;
    ++*(_QWORD *)(v2 + 32);
  }
  return v2;
}
