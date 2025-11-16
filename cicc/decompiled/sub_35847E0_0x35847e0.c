// Function: sub_35847E0
// Address: 0x35847e0
//
__int64 __fastcall sub_35847E0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned __int8 v8; // dl
  __int64 *v9; // rax
  __int64 v10; // rax
  int v11; // eax

  if ( *(_WORD *)(a2 + 68) == 24 )
  {
    v3 = *(_QWORD **)(a2 + 32);
    v4 = v3[8];
    v5 = v3[13];
    v6 = v3[18];
    v7 = sub_B10CD0(a2 + 56);
    if ( v7
      && ((v8 = *(_BYTE *)(v7 - 16), (v8 & 2) == 0)
        ? (v9 = (__int64 *)(v7 - 16 - 8LL * ((v8 >> 2) & 0xF)))
        : (v9 = *(__int64 **)(v7 - 32)),
          v10 = *v9,
          *(_BYTE *)v10 == 20) )
    {
      v11 = *(_DWORD *)(v10 + 4);
    }
    else
    {
      v11 = 0;
    }
    *(_DWORD *)a1 = v4;
    *(_DWORD *)(a1 + 4) = v5;
    *(_DWORD *)(a1 + 8) = v6;
    *(_DWORD *)(a1 + 12) = v11;
    *(_DWORD *)(a1 + 16) = 1065353216;
    *(_BYTE *)(a1 + 20) = 1;
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 20) = 0;
    return a1;
  }
}
