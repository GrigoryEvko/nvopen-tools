// Function: sub_2ECEFB0
// Address: 0x2ecefb0
//
__int64 __fastcall sub_2ECEFB0(__int64 a1)
{
  _QWORD **v1; // r12
  __int64 v2; // rax
  __int64 v3; // rax
  _BYTE *v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // r8
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_BYTE *)(a1 + 160) )
    sub_2ECEEC0(a1);
  v1 = *(_QWORD ***)(a1 + 64);
  if ( v1 == *(_QWORD ***)(a1 + 72) )
    goto LABEL_20;
  do
  {
    while ( !(unsigned __int8)sub_2ECEA00(a1, *v1) )
    {
      v2 = *(_QWORD *)(a1 + 72);
      if ( ++v1 == (_QWORD **)v2 )
        goto LABEL_12;
    }
    v3 = (__int64)*v1;
    v4 = *(_BYTE **)(a1 + 136);
    v8[0] = *v1;
    if ( v4 == *(_BYTE **)(a1 + 144) )
    {
      sub_2ECAD30(a1 + 128, v4, v8);
      v3 = v8[0];
    }
    else
    {
      if ( v4 )
      {
        *(_QWORD *)v4 = v3;
        v4 = *(_BYTE **)(a1 + 136);
      }
      *(_QWORD *)(a1 + 136) = v4 + 8;
    }
    *(_DWORD *)(v3 + 204) |= *(_DWORD *)(a1 + 88);
    *((_DWORD *)*v1 + 51) &= ~*(_DWORD *)(a1 + 24);
    *v1 = *(_QWORD **)(*(_QWORD *)(a1 + 72) - 8LL);
    v2 = *(_QWORD *)(a1 + 72) - 8LL;
    v1 = (_QWORD **)(*(_QWORD *)(a1 + 64) + (((unsigned __int64)v1 - *(_QWORD *)(a1 + 64)) & 0x7FFFFFFF8LL));
    *(_QWORD *)(a1 + 72) = v2;
  }
  while ( v1 != (_QWORD **)v2 );
LABEL_12:
  v5 = *(_QWORD *)(a1 + 64);
  if ( v5 == v2 )
  {
LABEL_20:
    do
    {
      sub_2EC8DA0(a1, *(_DWORD *)(a1 + 164) + 1);
      sub_2ECEEC0(a1);
      v2 = *(_QWORD *)(a1 + 72);
      v5 = *(_QWORD *)(a1 + 64);
    }
    while ( v2 == v5 );
  }
  v6 = 0;
  if ( (unsigned int)((v2 - v5) >> 3) == 1 )
    return *(_QWORD *)v5;
  return v6;
}
