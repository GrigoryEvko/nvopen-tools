// Function: sub_1981660
// Address: 0x1981660
//
__int64 __fastcall sub_1981660(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  __int64 v9; // r14
  __int64 v10; // rax

  v7 = sub_146F1B0(*a2, a4);
  if ( sub_14562D0(v7) )
    goto LABEL_2;
  v9 = sub_146F1B0(*a2, a5);
  if ( sub_14562D0(v9) )
    goto LABEL_2;
  if ( sub_146CEE0(*a2, v7, a2[2]) )
  {
    a3 = sub_15FF5D0(a3);
    v10 = v7;
    v7 = v9;
    v9 = v10;
  }
  if ( *(_WORD *)(v7 + 24) != 7 || a2[2] != *(_QWORD *)(v7 + 48) )
  {
LABEL_2:
    *(_BYTE *)(a1 + 24) = 0;
  }
  else
  {
    *(_BYTE *)(a1 + 24) = 1;
    *(_DWORD *)a1 = a3;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = v9;
  }
  return a1;
}
