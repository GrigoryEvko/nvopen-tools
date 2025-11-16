// Function: sub_923000
// Address: 0x923000
//
__int64 __fastcall sub_923000(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char v5; // dl
  unsigned __int64 v6; // rax
  bool v8; // al
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax

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
    v8 = sub_91B770(a3);
    v10 = *(_QWORD *)(a2 + 32) + 8LL;
    if ( v8 )
    {
      v11 = sub_91A390(v10, a3, 0, v9);
      v12 = sub_BCE760(v11, 0);
      if ( *(char *)(a3 + 142) >= 0 && *(_BYTE *)(a3 + 140) == 12 )
        v13 = sub_8D4AB0(a3);
      else
        v13 = *(_DWORD *)(a3 + 136);
      v14 = sub_ACA8A0(v12);
      *(_DWORD *)(a1 + 16) = v13;
      *(_QWORD *)a1 = v14;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_DWORD *)(a1 + 8) = 1;
      return a1;
    }
    else
    {
      v15 = sub_91A390(v10, a3, 0, v9);
      v16 = sub_ACA8A0(v15);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v16;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
  }
}
