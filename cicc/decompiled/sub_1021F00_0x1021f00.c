// Function: sub_1021F00
// Address: 0x1021f00
//
__int64 __fastcall sub_1021F00(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  char v9; // al
  _BYTE *v12; // rax
  __int64 *v13; // rdx
  _BYTE *v14; // rsi
  __int64 v15; // rcx
  _BYTE **v16; // rdx
  _BYTE *v17; // rax
  bool v18; // zf
  int v19; // edx

  v8 = *(_QWORD *)(a4 + 16);
  v9 = *(_BYTE *)a4;
  if ( !v8 || *(_QWORD *)(v8 + 8) || (unsigned __int8)(v9 - 82) > 1u )
  {
    if ( v9 != 86 )
      goto LABEL_3;
  }
  else
  {
    sub_B53900(a4);
    v12 = *(_BYTE **)(*(_QWORD *)(a4 + 16) + 24LL);
    if ( *v12 == 86 )
    {
      v19 = *(_DWORD *)(a5 + 16);
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = v12;
      *(_DWORD *)(a1 + 16) = v19;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    if ( *(_BYTE *)a4 != 86 )
      goto LABEL_3;
  }
  if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
    v13 = *(__int64 **)(a4 - 8);
  else
    v13 = (__int64 *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
  if ( (unsigned __int8)(*(_BYTE *)*v13 - 82) > 1u )
    goto LABEL_3;
  sub_B53900(*v13);
  v14 = *(_BYTE **)(a4 - 32);
  v15 = 0;
  if ( **(_BYTE **)(a4 - 64) == 84 )
    v15 = *(_QWORD *)(a4 - 64);
  if ( (_BYTE *)v15 != a3 )
  {
    if ( *v14 != 84 )
      v14 = 0;
    if ( v14 != a3 )
      goto LABEL_3;
    v14 = *(_BYTE **)(a4 - 64);
  }
  if ( !(unsigned __int8)sub_D48480(a2, (__int64)v14, 0, v15) )
  {
LABEL_3:
    *(_BYTE *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a4;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
    v16 = *(_BYTE ***)(a4 - 8);
  else
    v16 = (_BYTE **)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
  v17 = *v16;
  *(_BYTE *)a1 = 1;
  *(_QWORD *)(a1 + 8) = a4;
  v18 = *v17 == 82;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 16) = !v18 + 17;
  return a1;
}
