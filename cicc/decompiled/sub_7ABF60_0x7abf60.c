// Function: sub_7ABF60
// Address: 0x7abf60
//
__int64 __fastcall sub_7ABF60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rdx
  __int64 v5; // rax
  _BYTE v7[64]; // [rsp+0h] [rbp-40h] BYREF

  v2 = sub_87A100(a1, a2, v7);
  v3 = 0;
  if ( !v2 )
    return v3;
  v4 = v2;
  v5 = *(_QWORD *)(v2 + 32);
  if ( !unk_4D04968 )
    v5 = *(_QWORD *)(v4 + 24);
  if ( !v5 )
    return v3;
  do
  {
    if ( *(_BYTE *)(v5 + 80) == 1 )
      return 1;
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v5 );
  return 0;
}
