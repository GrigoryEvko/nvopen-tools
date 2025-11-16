// Function: sub_D793D0
// Address: 0xd793d0
//
void __fastcall sub_D793D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r12
  __int64 v7; // rdx

  v6 = *(_QWORD **)(a1 + 8);
  if ( v6 == *(_QWORD **)(a1 + 16) )
  {
    sub_9D3840((__int64 *)a1, *(char **)(a1 + 8), (__int64 *)a2);
  }
  else
  {
    if ( v6 )
    {
      *v6 = *(_QWORD *)a2;
      v6[1] = v6 + 3;
      v6[2] = 0xC00000000LL;
      v7 = *(unsigned int *)(a2 + 16);
      if ( (_DWORD)v7 )
        sub_D768F0((__int64)(v6 + 1), (char **)(a2 + 8), v7, a4, a5, a6);
      v6[9] = v6 + 11;
      v6[10] = 0xC00000000LL;
      if ( *(_DWORD *)(a2 + 80) )
        sub_D768F0((__int64)(v6 + 9), (char **)(a2 + 72), v7, a4, a5, a6);
      v6 = *(_QWORD **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v6 + 17;
  }
}
