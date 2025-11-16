// Function: sub_254CBC0
// Address: 0x254cbc0
//
void __fastcall sub_254CBC0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int v5; // eax
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rdx

  v2 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  if ( (sub_B2FC80((__int64)v2) || (unsigned __int8)sub_B2FC00(v2))
    && !(unsigned __int8)sub_B19060(*(_QWORD *)(a2 + 208) + 248LL, (__int64)v2, v3, v4)
    && (!*(_QWORD *)(a2 + 4432)
     || !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int8 *))(a2 + 4440))(a2 + 4416, v2)) )
  {
    if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v5 = *(_DWORD *)(a1 + 144), v5 <= 0x40) )
    {
      v7 = *(_QWORD *)(a1 + 136);
      *(_DWORD *)(a1 + 112) = v5;
      *(_QWORD *)(a1 + 104) = v7;
    }
    else
    {
      sub_C43990(a1 + 104, a1 + 136);
    }
    if ( *(_DWORD *)(a1 + 128) <= 0x40u && (v6 = *(_DWORD *)(a1 + 160), v6 <= 0x40) )
    {
      v8 = *(_QWORD *)(a1 + 152);
      *(_DWORD *)(a1 + 128) = v6;
      *(_QWORD *)(a1 + 120) = v8;
    }
    else
    {
      sub_C43990(a1 + 120, a1 + 152);
    }
  }
}
