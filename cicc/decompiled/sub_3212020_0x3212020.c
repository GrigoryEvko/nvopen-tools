// Function: sub_3212020
// Address: 0x3212020
//
__int64 __fastcall sub_3212020(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 i; // rbx
  unsigned __int64 v4; // rax
  unsigned __int8 v6; // al
  __int64 v7; // rdx

  v1 = 0x4440050002000201LL;
  v2 = a1;
  for ( i = a1; ; i = v2 )
  {
    v4 = (unsigned int)sub_AF18C0(v2) - 13;
    if ( (unsigned __int16)v4 > 0x3Eu || !_bittest64(&v1, v4) )
      return *(_QWORD *)(i + 24);
    if ( *(_BYTE *)i != 13 && *(_BYTE *)i != 36 )
      break;
    v6 = *(_BYTE *)(i - 16);
    v7 = (v6 & 2) != 0 ? *(_QWORD *)(i - 32) : i - 16 - 8LL * ((v6 >> 2) & 0xF);
    v2 = *(_QWORD *)(v7 + 24);
    if ( !v2 )
      break;
    if ( (unsigned __int16)sub_AF18C0(*(_QWORD *)(v7 + 24)) == 16 || (unsigned __int16)sub_AF18C0(v2) == 66 )
      return *(_QWORD *)(i + 24);
  }
  return 0;
}
