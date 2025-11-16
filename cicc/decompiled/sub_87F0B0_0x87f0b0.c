// Function: sub_87F0B0
// Address: 0x87f0b0
//
void __fastcall sub_87F0B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _QWORD *v6; // r14
  __int64 v7; // rdx

  v2 = *a2;
  if ( *a2 )
  {
    if ( *(_BYTE *)(v2 + 80) == 17 )
    {
      *(_QWORD *)(a1 + 8) = *(_QWORD *)(v2 + 88);
      *(_QWORD *)(v2 + 88) = a1;
      *(_BYTE *)(a1 + 83) |= 0x20u;
    }
    else
    {
      v6 = sub_87EBB0(0x11u, *(_QWORD *)v2, (_QWORD *)(v2 + 48));
      *((_DWORD *)v6 + 10) = *(_DWORD *)(v2 + 40);
      *((_DWORD *)v6 + 11) = *(_DWORD *)(v2 + 44);
      *((_BYTE *)v6 + 84) = *(_BYTE *)(v2 + 84) & 4 | *((_BYTE *)v6 + 84) & 0xFB;
      v7 = *(_QWORD *)(v2 + 64);
      if ( (*(_BYTE *)(v2 + 81) & 0x10) != 0 )
      {
        sub_877E20((__int64)v6, 0, v7, v3, v4, v5);
        *(_QWORD *)(a1 + 8) = v2;
        v6[11] = a1;
        *a2 = (__int64)v6;
      }
      else
      {
        if ( v7 )
          sub_877E90((__int64)v6, 0, v7);
        *(_QWORD *)(a1 + 8) = v2;
        v6[11] = a1;
        *a2 = (__int64)v6;
      }
    }
  }
  else
  {
    *a2 = a1;
  }
}
