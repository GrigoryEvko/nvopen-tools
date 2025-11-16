// Function: sub_B157E0
// Address: 0xb157e0
//
void __fastcall sub_B157E0(__int64 a1, _QWORD *a2)
{
  bool v2; // zf
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  _BYTE **v5; // rax
  _BYTE *v6; // rax
  unsigned __int8 v7; // dl
  _BYTE **v8; // rax

  v2 = *a2 == 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  if ( !v2 )
  {
    v3 = sub_B10CD0((__int64)a2);
    v4 = *(_BYTE *)(v3 - 16);
    if ( (v4 & 2) != 0 )
      v5 = *(_BYTE ***)(v3 - 32);
    else
      v5 = (_BYTE **)(v3 - 16 - 8LL * ((v4 >> 2) & 0xF));
    v6 = *v5;
    if ( *v6 != 16 )
    {
      v7 = *(v6 - 16);
      if ( (v7 & 2) != 0 )
        v8 = (_BYTE **)*((_QWORD *)v6 - 4);
      else
        v8 = (_BYTE **)&v6[-8 * ((v7 >> 2) & 0xF) - 16];
      v6 = *v8;
    }
    *(_QWORD *)a1 = v6;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(sub_B10CD0((__int64)a2) + 4);
    *(_DWORD *)(a1 + 12) = *(unsigned __int16 *)(sub_B10CD0((__int64)a2) + 2);
  }
}
