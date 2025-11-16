// Function: sub_1E756D0
// Address: 0x1e756d0
//
__int64 __fastcall sub_1E756D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v6; // rax
  _BYTE *v7; // rsi
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_BYTE *)(a1 + 160) )
    sub_1E75380(a1);
  v1 = *(_QWORD *)(a1 + 72);
  if ( !*(_DWORD *)(a1 + 168) )
    goto LABEL_16;
  v2 = *(__int64 **)(a1 + 64);
  if ( v2 != (__int64 *)v1 )
  {
    do
    {
      while ( !(unsigned __int8)sub_1E72C10(a1, *v2) )
      {
        v1 = *(_QWORD *)(a1 + 72);
        if ( ++v2 == (__int64 *)v1 )
          goto LABEL_16;
      }
      v6 = *v2;
      v7 = *(_BYTE **)(a1 + 136);
      v8[0] = *v2;
      if ( v7 == *(_BYTE **)(a1 + 144) )
      {
        sub_1CFD630(a1 + 128, v7, v8);
        v6 = v8[0];
      }
      else
      {
        if ( v7 )
        {
          *(_QWORD *)v7 = v6;
          v7 = *(_BYTE **)(a1 + 136);
        }
        *(_QWORD *)(a1 + 136) = v7 + 8;
      }
      *(_DWORD *)(v6 + 196) |= *(_DWORD *)(a1 + 88);
      *(_DWORD *)(*v2 + 196) &= ~*(_DWORD *)(a1 + 24);
      *v2 = *(_QWORD *)(*(_QWORD *)(a1 + 72) - 8LL);
      v1 = *(_QWORD *)(a1 + 72) - 8LL;
      v2 = (__int64 *)(*(_QWORD *)(a1 + 64) + (((unsigned __int64)v2 - *(_QWORD *)(a1 + 64)) & 0x7FFFFFFF8LL));
      *(_QWORD *)(a1 + 72) = v1;
    }
    while ( v2 != (__int64 *)v1 );
LABEL_16:
    v3 = *(_QWORD *)(a1 + 64);
    if ( v3 != v1 )
      goto LABEL_6;
    goto LABEL_5;
  }
  do
  {
LABEL_5:
    sub_1E72EF0(a1, *(_DWORD *)(a1 + 164) + 1);
    sub_1E75380(a1);
    v1 = *(_QWORD *)(a1 + 72);
    v3 = *(_QWORD *)(a1 + 64);
  }
  while ( v1 == v3 );
LABEL_6:
  v4 = 0;
  if ( (unsigned int)((v1 - v3) >> 3) == 1 )
    return *(_QWORD *)v3;
  return v4;
}
