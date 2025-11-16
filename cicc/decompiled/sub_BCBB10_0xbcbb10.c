// Function: sub_BCBB10
// Address: 0xbcbb10
//
bool __fastcall sub_BCBB10(__int64 a1, unsigned __int8 *a2)
{
  _BYTE *v2; // r12
  __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // rdx
  _BYTE *v7; // rax
  _QWORD *v8; // rax

  v2 = a2;
  v4 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( !sub_BCAC40(v4, 32) )
    return 0;
  v5 = *(unsigned __int8 *)(*((_QWORD *)a2 + 1) + 8LL);
  if ( (_BYTE)v5 == 18 )
    return 0;
  v6 = *a2;
  if ( (unsigned __int8)v6 > 0x15u )
    return 0;
  if ( (unsigned int)(v5 - 17) <= 1 )
  {
    v7 = sub_AD7630((__int64)a2, 0, v6);
    v2 = v7;
    if ( !v7 )
      return 0;
    LOBYTE(v6) = *v7;
  }
  if ( (_BYTE)v6 != 17 )
    return 0;
  v8 = (_QWORD *)*((_QWORD *)v2 + 3);
  if ( *((_DWORD *)v2 + 8) > 0x40u )
    v8 = (_QWORD *)*v8;
  return *(unsigned int *)(a1 + 12) > (unsigned __int64)v8;
}
