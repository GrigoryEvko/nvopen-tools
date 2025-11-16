// Function: sub_8540F0
// Address: 0x8540f0
//
__int64 __fastcall sub_8540F0(_QWORD *a1, char a2, __int64 a3, int a4)
{
  int v5; // r15d
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  int v19[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = dword_4F04C64;
  if ( dword_4F04C64 != -1 )
  {
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(result + 14) & 2) != 0 )
      return result;
  }
  if ( a3 )
  {
    if ( a2 == 21 )
    {
      *(_BYTE *)(a3 + 41) |= 1u;
      v7 = 0;
      goto LABEL_6;
    }
    v7 = sub_72A270(a3, a2);
    v12 = *(_BYTE *)(v7 + 89);
    if ( (v12 & 4) != 0 || (v13 = *(_QWORD *)(v7 + 40)) != 0 && *(_BYTE *)(v13 + 28) == 3 )
    {
      if ( dword_4F077C4 != 2 )
      {
LABEL_26:
        *(_BYTE *)(v7 + 88) |= 0x80u;
        v5 = 0;
        goto LABEL_7;
      }
    }
    else if ( (v12 & 1) != 0 )
    {
      if ( (*(_BYTE *)(v7 - 8) & 1) == 0 )
      {
        *(_BYTE *)(v7 + 88) |= 0x80u;
        goto LABEL_6;
      }
      goto LABEL_26;
    }
    *(_BYTE *)(v7 + 88) |= 0x80u;
    v5 = -1;
    v18 = v7;
    v15 = sub_727BE0(*(_BYTE *)(a1[1] + 8LL), (_QWORD *)v7);
    v9 = v18;
    v10 = v15;
    *(_QWORD *)(v15 + 32) = a1[7];
    *(_QWORD *)(v15 + 48) = a1[10];
    *(_BYTE *)(v15 + 9) = *(_BYTE *)(a1[1] + 18LL) & 1;
    goto LABEL_9;
  }
  if ( a4 )
  {
    v7 = 0;
    v5 = 0;
    goto LABEL_7;
  }
  v11 = 0;
  if ( dword_4F04C64 != -1 )
    v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v11 + 4) )
    {
      case 0:
      case 2:
      case 3:
      case 4:
      case 0x10:
      case 0x11:
        goto LABEL_30;
      case 1:
      case 5:
      case 7:
      case 8:
      case 9:
      case 0xD:
      case 0xE:
      case 0xF:
        v14 = *(_DWORD *)(v11 + 552);
        v11 = 0;
        if ( v14 != -1 )
          goto LABEL_28;
        continue;
      case 6:
        v14 = *(_DWORD *)(v11 + 552);
        if ( dword_4F077C4 != 2 )
        {
          v11 = 0;
          if ( v14 != -1 )
          {
LABEL_28:
            v11 = qword_4F04C68[0] + 776LL * v14;
            if ( *(_BYTE *)(v11 + 4) > 0x11u )
LABEL_29:
              sub_721090();
          }
          continue;
        }
LABEL_30:
        v7 = 0;
        v5 = 1594008481 * ((v11 - qword_4F04C68[0]) >> 3);
LABEL_6:
        if ( v5 != -1 )
        {
LABEL_7:
          v16 = v7;
          sub_7296F0(v5, v19);
          v7 = v16;
        }
        v17 = v7;
        v8 = sub_727BE0(*(_BYTE *)(a1[1] + 8LL), (_QWORD *)v7);
        v9 = v17;
        v10 = v8;
        *(_QWORD *)(v8 + 32) = a1[7];
        *(_QWORD *)(v8 + 48) = a1[10];
        *(_BYTE *)(v8 + 9) = *(_BYTE *)(a1[1] + 18LL) & 1;
        if ( a3 )
        {
LABEL_9:
          *(_BYTE *)(v10 + 16) = a2;
          *(_QWORD *)(v10 + 24) = a3;
        }
        sub_7363B0(v10, v5, v9);
        if ( v5 != -1 )
          sub_729730(v19[0]);
        result = dword_4F04C3C;
        if ( !dword_4F04C3C )
          result = sub_8699D0(v10, 58, a1[8]);
        a1[11] = v10;
        a1[8] = 0;
        return result;
      default:
        goto LABEL_29;
    }
  }
}
