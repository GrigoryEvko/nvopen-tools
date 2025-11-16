// Function: sub_8291E0
// Address: 0x8291e0
//
__int64 __fastcall sub_8291E0(__int64 a1)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax
  __int64 v4; // rdx

  if ( *(_BYTE *)(a1 + 174) == 7
    && (*(_BYTE *)(a1 + 193) & 0x10) != 0
    && (v2 = *(_QWORD *)(a1 + 152), (v3 = **(_QWORD ***)(v2 + 168)) != 0)
    && ((v4 = *(_QWORD *)(v2 + 160), *v3) || v4 != v3[1]) )
  {
    return sub_8D3BB0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 168) + 160LL) + 192LL));
  }
  else
  {
    return 0;
  }
}
