// Function: sub_1280E20
// Address: 0x1280e20
//
__int64 __fastcall sub_1280E20(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char v5; // dl
  unsigned __int64 v6; // rax
  bool v8; // al
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r14
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax

  v5 = *(_BYTE *)(a3 + 140);
  if ( v5 == 12 )
  {
    v6 = a3;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v5 = *(_BYTE *)(v6 + 140);
    }
    while ( v5 == 12 );
  }
  if ( v5 == 1 )
  {
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    v8 = sub_127B420(a3);
    v9 = *(_QWORD *)(a2 + 32) + 8LL;
    if ( v8 )
    {
      v10 = sub_127A030(v9, a3, 0);
      v11 = sub_1646BA0(v10, 0);
      if ( *(char *)(a3 + 142) >= 0 && *(_BYTE *)(a3 + 140) == 12 )
        v12 = sub_8D4AB0(a3);
      else
        v12 = *(_DWORD *)(a3 + 136);
      v13 = sub_1599EF0(v11);
      *(_DWORD *)(a1 + 16) = v12;
      *(_QWORD *)a1 = v13;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_DWORD *)(a1 + 8) = 1;
      return a1;
    }
    else
    {
      v14 = sub_127A030(v9, a3, 0);
      v15 = sub_1599EF0(v14);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v15;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
  }
}
