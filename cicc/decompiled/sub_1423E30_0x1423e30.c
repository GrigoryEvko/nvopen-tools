// Function: sub_1423E30
// Address: 0x1423e30
//
unsigned __int64 __fastcall sub_1423E30(_QWORD *a1, _DWORD *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v5; // rax
  __int128 v6; // [rsp+0h] [rbp-A0h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v5 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v5 = 0xFF51AFD7ED558CCDLL;
    unk_4F99938 = v5;
    sub_2207640(byte_4F99930);
  }
  *(_QWORD *)&v6 = *a1;
  DWORD2(v6) = *a2;
  v2 = __ROR8__(*(_QWORD *)((char *)&v6 + 4) + 12LL, 12);
  v3 = 0x9DDFEA08EB382D69LL
     * (((0x9DDFEA08EB382D69LL * (*a1 ^ v2 ^ unk_4F99938)) >> 47)
      ^ (0x9DDFEA08EB382D69LL * (*a1 ^ v2 ^ unk_4F99938))
      ^ v2);
  return *(_QWORD *)((char *)&v6 + 4) ^ (0x9DDFEA08EB382D69LL * ((v3 >> 47) ^ v3));
}
