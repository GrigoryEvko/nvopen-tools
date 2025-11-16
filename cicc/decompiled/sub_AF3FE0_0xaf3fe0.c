// Function: sub_AF3FE0
// Address: 0xaf3fe0
//
__int64 __fastcall sub_AF3FE0(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rdi
  char *v3; // rax
  unsigned __int8 v4; // cl
  unsigned __int8 v5; // dl
  char *v6; // rax
  __int64 v8; // [rsp+0h] [rbp-10h]

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(_QWORD *)(a1 - 32);
  else
    v2 = a1 - 16 - 8LL * ((v1 >> 2) & 0xF);
  v3 = *(char **)(v2 + 24);
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = *v3;
      if ( (unsigned __int8)*v3 <= 0x24u && ((1LL << v4) & 0x140000F000LL) != 0 && *((_QWORD *)v3 + 3) )
        break;
      if ( v4 == 13 )
      {
        v5 = *(v3 - 16);
        v6 = (v5 & 2) != 0 ? (char *)*((_QWORD *)v3 - 4) : &v3[-8 * ((v5 >> 2) & 0xF) - 16];
        v3 = (char *)*((_QWORD *)v6 + 3);
        if ( v3 )
          continue;
      }
      return v8;
    }
    return *((_QWORD *)v3 + 3);
  }
  return v8;
}
