// Function: sub_E92200
// Address: 0xe92200
//
__int64 __fastcall sub_E92200(__int64 a1, __int64 a2, char a3)
{
  _DWORD *v3; // r8
  int v4; // eax
  __int64 v5; // r12
  _DWORD *v6; // rax
  int v8; // eax
  _DWORD v9[2]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v10; // [rsp+8h] [rbp-18h]

  v3 = *(_DWORD **)(a1 + 152);
  v4 = *(_DWORD *)(a1 + 124);
  if ( !a3 )
  {
    v3 = *(_DWORD **)(a1 + 144);
    v4 = *(_DWORD *)(a1 + 120);
  }
  if ( v3
    && (v9[0] = a2, v5 = (__int64)&v3[2 * v4], v9[1] = 0, v6 = sub_E92140(v3, v5, v9), (_DWORD *)v5 != v6)
    && *v6 == a2 )
  {
    v8 = v6[1];
    BYTE4(v10) = 1;
    LODWORD(v10) = v8;
    return v10;
  }
  else
  {
    BYTE4(v10) = 0;
    return v10;
  }
}
