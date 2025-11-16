// Function: sub_1423D60
// Address: 0x1423d60
//
unsigned __int64 __fastcall sub_1423D60(_BYTE *a1, _DWORD *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // [rsp+0h] [rbp-A0h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v5 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v5 = 0xFF51AFD7ED558CCDLL;
    unk_4F99938 = v5;
    sub_2207640(byte_4F99930);
  }
  *(_DWORD *)((char *)&v6 + 1) = *a2;
  v2 = unk_4F99938 ^ (unsigned int)*a2;
  LOBYTE(v6) = *a1;
  v3 = 0x9DDFEA08EB382D69LL * (v2 ^ (8LL * (unsigned int)v6 + 5));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v3 ^ v2 ^ (v3 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v3 ^ v2 ^ (v3 >> 47))));
}
