// Function: sub_31F7430
// Address: 0x31f7430
//
__int64 __fastcall sub_31F7430(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v2; // al
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdx

  v1 = a1 - 16;
  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 16LL);
    if ( !v4 )
    {
LABEL_11:
      v7 = *(_QWORD *)(a1 - 32);
      goto LABEL_7;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 - 8LL * ((v2 >> 2) & 0xF));
    if ( !v4 )
      goto LABEL_6;
  }
  sub_B91420(v4);
  if ( v5 )
    return 0;
  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
    goto LABEL_11;
LABEL_6:
  v7 = v1 - 8LL * ((v2 >> 2) & 0xF);
LABEL_7:
  v8 = *(_QWORD *)(v7 + 56);
  if ( v8 )
  {
    sub_B91420(v8);
    if ( v9 )
      return 0;
  }
  return ((unsigned __int8)(*(_DWORD *)(a1 + 20) >> 2) ^ 1) & 1;
}
