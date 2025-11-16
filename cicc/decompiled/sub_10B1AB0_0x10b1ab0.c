// Function: sub_10B1AB0
// Address: 0x10b1ab0
//
__int64 __fastcall sub_10B1AB0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 || *(_QWORD *)(v6 + 8) || *(_BYTE *)v5 != 57 )
    return 0;
  v7 = *(_QWORD *)(v5 - 64);
  if ( !v7 )
    goto LABEL_15;
  **a1 = v7;
  v8 = *(_QWORD *)(v5 - 32);
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) )
  {
LABEL_9:
    **a1 = v8;
    v10 = *(_QWORD *)(v5 - 64);
    v11 = *(_QWORD *)(v10 + 16);
    if ( v11 && !*(_QWORD *)(v11 + 8) && sub_10B14D0(a1 + 1, 15, (unsigned __int8 *)v10) )
      return sub_995B10(a1 + 3, *((_QWORD *)a3 - 4));
    return 0;
  }
  if ( !sub_10B14D0(a1 + 1, 15, (unsigned __int8 *)v8) )
  {
LABEL_15:
    v8 = *(_QWORD *)(v5 - 32);
    if ( !v8 )
      return 0;
    goto LABEL_9;
  }
  return sub_995B10(a1 + 3, *((_QWORD *)a3 - 4));
}
