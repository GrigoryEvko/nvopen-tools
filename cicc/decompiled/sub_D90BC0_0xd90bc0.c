// Function: sub_D90BC0
// Address: 0xd90bc0
//
char __fastcall sub_D90BC0(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // rcx
  char v3; // r8
  __int64 v5; // rsi

  v1 = *a1;
  if ( (unsigned __int8)(v1 - 42) > 0x33u )
    return 0;
  v2 = 0x8133FFE2BFFFFLL;
  v3 = 1;
  if ( _bittest64(&v2, (unsigned int)(v1 - 42)) )
    return v3;
  if ( (_BYTE)v1 != 85 )
    return 0;
  v5 = *((_QWORD *)a1 - 4);
  if ( !v5 )
    return 0;
  v3 = 0;
  if ( *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *((_QWORD *)a1 + 10) )
    return v3;
  return sub_971E80((__int64)a1, v5);
}
