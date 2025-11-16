// Function: sub_6ECD90
// Address: 0x6ecd90
//
__int64 __fastcall sub_6ECD90(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int64 v2; // rtt
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // r15
  unsigned int v6; // [rsp+Ch] [rbp-44h]
  unsigned int v7; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = unk_4D048C0;
  if ( unk_4D048C0 )
  {
    v1 = dword_4D04964;
    if ( dword_4D04964 )
    {
      return 0;
    }
    else if ( *(_BYTE *)(a1 + 137) )
    {
      v2 = *(unsigned __int8 *)(a1 + 137);
      v3 = v2 % dword_4F06BA0;
      v4 = v2 / dword_4F06BA0;
      if ( !v3 && !*(_BYTE *)(a1 + 136) )
      {
        v6 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 136LL);
        while ( 1 )
        {
          if ( byte_4B6DF90[v3] == ((*(_BYTE *)(a1 + 144) & 8) != 0) )
          {
            sub_622920((unsigned int)v3, v8, &v7);
            if ( v4 == v8[0] && v6 >= v7 && !(*(_QWORD *)(a1 + 128) % (unsigned __int64)v7) )
              break;
          }
          if ( ++v3 == 13 )
            return v1;
        }
        return 1;
      }
    }
  }
  return v1;
}
