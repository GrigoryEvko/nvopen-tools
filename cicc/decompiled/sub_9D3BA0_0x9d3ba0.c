// Function: sub_9D3BA0
// Address: 0x9d3ba0
//
void __fastcall sub_9D3BA0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12

  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 == *(_QWORD **)(a1 + 16) )
  {
    sub_9D3840((__int64 *)a1, *(char **)(a1 + 8), (__int64 *)a2);
  }
  else
  {
    if ( v2 )
    {
      *v2 = *(_QWORD *)a2;
      v2[1] = v2 + 3;
      v2[2] = 0xC00000000LL;
      if ( *(_DWORD *)(a2 + 16) )
        sub_9C31C0((__int64)(v2 + 1), (char **)(a2 + 8));
      v2[9] = v2 + 11;
      v2[10] = 0xC00000000LL;
      if ( *(_DWORD *)(a2 + 80) )
        sub_9C31C0((__int64)(v2 + 9), (char **)(a2 + 72));
      v2 = *(_QWORD **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v2 + 17;
  }
}
