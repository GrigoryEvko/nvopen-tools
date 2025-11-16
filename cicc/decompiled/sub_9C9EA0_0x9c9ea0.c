// Function: sub_9C9EA0
// Address: 0x9c9ea0
//
void __fastcall sub_9C9EA0(__int64 *a1, __int64 *a2, _DWORD *a3)
{
  __int64 v4; // r12
  __int64 v5; // rax

  v4 = a1[1];
  if ( v4 == a1[2] )
  {
    sub_9C9BE0(a1, a1[1], a2, a3);
  }
  else
  {
    if ( v4 )
    {
      v5 = *a2;
      *(_QWORD *)v4 = 6;
      *(_QWORD *)(v4 + 8) = 0;
      *(_QWORD *)(v4 + 16) = v5;
      if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
        sub_BD73F0(v4);
      *(_DWORD *)(v4 + 24) = *a3;
      v4 = a1[1];
    }
    a1[1] = v4 + 32;
  }
}
