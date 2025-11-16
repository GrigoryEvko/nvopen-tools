// Function: sub_D5D4E0
// Address: 0xd5d4e0
//
char __fastcall sub_D5D4E0(__int64 a1, int a2)
{
  int v2; // edx
  __int64 v3; // rdx
  __int64 v4; // rax
  _QWORD *v5; // rcx
  __int64 v7; // [rsp+8h] [rbp-28h] BYREF
  __int64 v8; // [rsp+14h] [rbp-1Ch]
  int v9; // [rsp+1Ch] [rbp-14h]

  v8 = sub_D5D290(a1, a2);
  v9 = v2;
  if ( (_BYTE)v2 )
  {
    v3 = *(_QWORD *)(a1 + 24);
    LOBYTE(v4) = 0;
    v5 = *(_QWORD **)(v3 + 16);
    if ( *(_BYTE *)(*v5 + 8LL) == 7 && (_DWORD)v8 == *(_DWORD *)(v3 + 12) - 1 )
      LOBYTE(v4) = *(_BYTE *)(v5[1] + 8LL) == 14;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 120);
    return ((unsigned __int64)sub_A746D0(&v7) >> 2) & 1;
  }
  return v4;
}
