// Function: sub_C1B090
// Address: 0xc1b090
//
__int64 __fastcall sub_C1B090(__int64 a1, char a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rcx
  __int64 v4; // rcx
  __int64 result; // rax
  unsigned int v6; // edx
  _QWORD *v7; // rcx
  _BYTE *v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rcx
  __int64 v11; // rcx
  unsigned int v12; // eax

  v2 = *(_BYTE *)(a1 - 16);
  v3 = a1 - 16;
  if ( !unk_4F838D4 )
  {
    if ( a2 )
    {
      if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
      {
        v10 = *(_QWORD **)(a1 - 32);
        v8 = (_BYTE *)*v10;
        if ( *(_BYTE *)*v10 == 20 )
          goto LABEL_12;
      }
      else
      {
        v7 = (_QWORD *)(v3 - 8LL * ((v2 >> 2) & 0xF));
        v8 = (_BYTE *)*v7;
        if ( *(_BYTE *)*v7 == 20 )
        {
LABEL_12:
          v9 = *((unsigned int *)v8 + 1);
          return (v9 << 32) | (unsigned int)sub_C1B040(a1);
        }
      }
    }
    else
    {
      if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
        v11 = *(_QWORD *)(a1 - 32);
      else
        v11 = v3 - 8LL * ((v2 >> 2) & 0xF);
      v9 = 0;
      if ( **(_BYTE **)v11 != 20 )
        return (v9 << 32) | (unsigned int)sub_C1B040(a1);
      v12 = *(_DWORD *)(*(_QWORD *)v11 + 4LL);
      if ( (v12 & 7) == 7 && (v12 & 0xFFFFFFF8) != 0 )
      {
        if ( (v12 & 0x10000000) != 0 )
          v9 = HIWORD(v12) & 7;
        else
          v9 = (unsigned __int16)(v12 >> 3);
        return (v9 << 32) | (unsigned int)sub_C1B040(a1);
      }
      v9 = (unsigned __int8)v12;
      if ( LOBYTE(qword_4F813A8[8]) )
        return (v9 << 32) | (unsigned int)sub_C1B040(a1);
      if ( (v12 & 1) == 0 )
      {
        v9 = (v12 >> 1) & 0x1F;
        if ( ((v12 >> 1) & 0x20) != 0 )
          v9 = (v12 >> 2) & 0xFE0 | (unsigned int)v9;
        return (v9 << 32) | (unsigned int)sub_C1B040(a1);
      }
    }
    v9 = 0;
    return (v9 << 32) | (unsigned int)sub_C1B040(a1);
  }
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v4 = *(_QWORD *)(a1 - 32);
  else
    v4 = v3 - 8LL * ((v2 >> 2) & 0xF);
  LOWORD(result) = 0;
  if ( **(_BYTE **)v4 == 20 )
  {
    v6 = *(_DWORD *)(*(_QWORD *)v4 + 4LL);
    LOWORD(result) = (unsigned __int16)v6 >> 3;
    if ( (v6 & 0x10000000) == 0 )
      LOWORD(result) = v6 >> 3;
  }
  return (unsigned __int16)result;
}
