// Function: sub_318EE30
// Address: 0x318ee30
//
__int64 __fastcall sub_318EE30(__int64 a1, unsigned __int8 *a2, _QWORD *a3)
{
  int v3; // eax
  int v6; // edx
  __int64 v7; // r12
  __int64 v8; // r9
  int v9; // ebx
  __int64 v10; // r10
  __int64 v11; // rdi
  int v12; // esi
  char v13; // r11
  unsigned int i; // eax
  unsigned __int8 *v15; // rsi
  unsigned int v16; // r8d
  __int64 v17; // r13
  unsigned int v18; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v3 - 1;
  v7 = 0;
  v8 = *((_QWORD *)a2 + 6);
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *((_QWORD *)a2 + 3);
  v12 = *a2;
  v13 = v12;
  for ( i = (v3 - 1) & (v12 ^ v11 ^ v8); ; i = v6 & v18 )
  {
    v15 = (unsigned __int8 *)(v10 + 72LL * i);
    v16 = *v15;
    v17 = *((_QWORD *)v15 + 3);
    if ( v13 == (_BYTE)v16 && v11 == v17 && v8 == *((_QWORD *)v15 + 6) )
    {
      *a3 = v15;
      return 1;
    }
    if ( !(_BYTE)v16 )
      break;
    if ( !v17 && !(*((_QWORD *)v15 + 6) | v7) )
      v7 = v10 + 72LL * i;
LABEL_7:
    v18 = v9 + i;
    ++v9;
  }
  if ( v17 || *((_QWORD *)v15 + 6) )
    goto LABEL_7;
  if ( !v7 )
    v7 = v10 + 72LL * i;
  *a3 = v7;
  return v16;
}
