// Function: sub_29191E0
// Address: 0x29191e0
//
char __fastcall sub_29191E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  char result; // al
  __int64 v6; // r13
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // r13d
  unsigned int v12; // r12d
  int v13; // ebx

  if ( a2 == a3 )
    return 1;
  v3 = a3;
  v4 = a2;
  if ( *(_BYTE *)(a2 + 8) == 12 && *(_BYTE *)(a3 + 8) == 12 )
    return *(_DWORD *)(a3 + 8) >> 8 <= 8u && *(_DWORD *)(a3 + 8) >> 8 > *(_DWORD *)(a2 + 8) >> 8;
  v6 = sub_9208B0(a1, a3);
  if ( v6 != sub_9208B0(a1, a2) )
    return 0;
  v7 = *(unsigned __int8 *)(v3 + 8);
  if ( (unsigned __int8)v7 > 3u && (_BYTE)v7 != 5 )
  {
    if ( (unsigned __int8)v7 > 0x14u )
      return 0;
    v9 = 1463376;
    if ( !_bittest64(&v9, v7) )
      return 0;
  }
  v8 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned __int8)v8 > 3u )
  {
    if ( (_BYTE)v8 == 5 )
      goto LABEL_11;
    if ( (unsigned __int8)v8 > 0x14u )
      return 0;
    v10 = 1463376;
    if ( !_bittest64(&v10, v8) )
      return 0;
  }
  if ( (unsigned int)(unsigned __int8)v8 - 17 <= 1 )
  {
    v4 = **(_QWORD **)(a2 + 16);
    LOBYTE(v8) = *(_BYTE *)(v4 + 8);
  }
LABEL_11:
  if ( (unsigned int)(unsigned __int8)v7 - 17 <= 1 )
  {
    v3 = **(_QWORD **)(v3 + 16);
    LOBYTE(v7) = *(_BYTE *)(v3 + 8);
  }
  if ( (_BYTE)v7 != 14 )
  {
    if ( (_BYTE)v8 != 14 )
      return (_BYTE)v7 != 20 && (_BYTE)v8 != 20;
    if ( *((_BYTE *)sub_AE2980(a1, *(_DWORD *)(v4 + 8) >> 8) + 16) )
      return 0;
    LOBYTE(v7) = *(_BYTE *)(v3 + 8);
    return (_BYTE)v7 == 12;
  }
  if ( (_BYTE)v8 != 14 )
  {
    if ( (_BYTE)v8 == 12 )
      return *((_BYTE *)sub_AE2980(a1, *(_DWORD *)(v3 + 8) >> 8) + 16) ^ 1;
    return (_BYTE)v7 == 12;
  }
  result = 1;
  v11 = *(_DWORD *)(v4 + 8) >> 8;
  v12 = *(_DWORD *)(v3 + 8) >> 8;
  if ( v12 != v11 )
  {
    if ( !*((_BYTE *)sub_AE2980(a1, v11) + 16) && !*((_BYTE *)sub_AE2980(a1, v12) + 16) )
    {
      v13 = sub_AE4380(a1, v11);
      return v13 == (unsigned int)sub_AE4380(a1, v12);
    }
    return 0;
  }
  return result;
}
