// Function: sub_878940
// Address: 0x878940
//
__int64 __fastcall sub_878940(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rax
  __int64 *v8; // rdx

  v2 = qword_4F5FFE8;
  if ( qword_4F5FFE8 )
    qword_4F5FFE8 = *(_QWORD *)qword_4F5FFE8;
  else
    v2 = sub_823970(56);
  *(_QWORD *)v2 = 0;
  *(_QWORD *)(v2 + 8) = a1;
  v3 = qword_4F04C68[0];
  *(_QWORD *)(v2 + 16) = a2;
  *(_QWORD *)(v2 + 24) = 0;
  *(_QWORD *)(v2 + 32) = 0;
  v4 = dword_4F04C64;
  *(_QWORD *)(v2 + 40) = 0;
  *(_DWORD *)(v2 + 48) = 0;
  *(_BYTE *)(v2 + 52) = 0;
  while ( 1 )
  {
    v5 = v3 + 776 * v4;
    if ( !v5 )
      goto LABEL_7;
    if ( *(_BYTE *)(v5 + 4) == 9 )
      break;
    v4 = *(int *)(v5 + 552);
    if ( (_DWORD)v4 == -1 )
      goto LABEL_7;
  }
  v6 = 1594008481 * ((v5 - v3) >> 3);
  if ( v6 == -1 )
LABEL_7:
    v6 = dword_4F04C44;
  v7 = v3 + 776LL * v6;
  if ( (*(_BYTE *)(v7 + 6) & 0xA) != 0 )
  {
LABEL_12:
    if ( *(_QWORD *)(v7 + 584) )
    {
      v8 = *(__int64 **)(v7 + 592);
      if ( !v8 )
      {
LABEL_15:
        *(_QWORD *)(v7 + 592) = v2;
        return v2;
      }
    }
    else
    {
      v8 = *(__int64 **)(v7 + 592);
      *(_QWORD *)(v7 + 584) = v2;
      if ( !v8 )
        goto LABEL_15;
    }
    *v8 = v2;
    goto LABEL_15;
  }
  if ( dword_4F04C44 != -1 )
  {
    v7 = v3 + 776LL * (int)dword_4F04C44;
    goto LABEL_12;
  }
  return v2;
}
