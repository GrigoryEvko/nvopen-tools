// Function: sub_7E6810
// Address: 0x7e6810
//
void __fastcall sub_7E6810(__int64 a1, __int64 a2, int a3)
{
  int v4; // r13d
  _QWORD *v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rdi

  v4 = *(_DWORD *)a2;
  if ( a3 )
  {
    v5 = *(_QWORD **)(a1 + 48);
    if ( v5 )
      sub_7E67B0(v5);
    if ( *(_BYTE *)(a1 + 40) == 13 )
    {
      v8 = *(_QWORD **)(*(_QWORD *)(a1 + 80) + 8LL);
      if ( v8 )
        sub_7E67B0(v8);
    }
  }
  if ( (unsigned int)(v4 - 3) <= 2 )
  {
    sub_7E25D0(*(_QWORD *)(a1 + 48), (int *)a2);
    return;
  }
  if ( (unsigned int)(v4 - 1) <= 1 && *(_BYTE *)(a2 + 24) )
  {
    *(_QWORD *)(a2 + 16) = a1;
    if ( v4 == 2 )
      goto LABEL_15;
  }
  else if ( v4 == 2 )
  {
    goto LABEL_15;
  }
  v6 = *(_QWORD **)(a2 + 8);
  if ( v4 != 1 )
  {
    *(_QWORD *)(a1 + 16) = v6[2];
    v6[2] = a1;
    *(_QWORD *)(a1 + 24) = v6[3];
    *(_QWORD *)(a2 + 8) = a1;
    if ( *(_BYTE *)(a1 + 40) == 17 )
      return;
LABEL_10:
    sub_7E2B60(*(_QWORD *)(a1 + 16));
    return;
  }
  v7 = v6[9];
  *(_QWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 16) = v7;
  v6[9] = a1;
LABEL_15:
  *(_DWORD *)a2 = 0;
  *(_QWORD *)(a2 + 8) = a1;
  if ( *(_BYTE *)(a1 + 40) != 17 )
    goto LABEL_10;
}
