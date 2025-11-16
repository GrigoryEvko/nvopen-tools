// Function: sub_223E060
// Address: 0x223e060
//
void __fastcall sub_223E060(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 v4; // rdi

  if ( (*(_BYTE *)(*(_QWORD *)(**(_QWORD **)(a1 + 8) - 24LL) + *(_QWORD *)(a1 + 8) + 25LL) & 0x20) != 0 )
  {
    v2 = *(_QWORD *)(**(_QWORD **)(a1 + 8) - 24LL) + *(_QWORD *)(a1 + 8);
    if ( !(unsigned __int8)sub_2252910() )
    {
      v3 = *(_QWORD *)(v2 + 232);
      if ( v3 )
      {
        if ( (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v3 + 48LL))(v3) == -1 )
        {
          v4 = *(_QWORD *)(**(_QWORD **)(a1 + 8) - 24LL) + *(_QWORD *)(a1 + 8);
          sub_222DC80(v4, *(_DWORD *)(v4 + 32) | 1);
        }
      }
    }
  }
}
