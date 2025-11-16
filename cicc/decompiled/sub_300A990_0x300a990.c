// Function: sub_300A990
// Address: 0x300a990
//
__int64 __fastcall sub_300A990(unsigned __int16 *a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 **v3; // r12
  int v4; // eax
  __int64 *v5; // r13
  __int64 v6; // rsi
  unsigned __int16 v7; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // r12
  char v10; // dl
  unsigned __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 **v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // [rsp+8h] [rbp-58h]
  unsigned __int64 v22; // [rsp+10h] [rbp-50h] BYREF
  __int64 **v23; // [rsp+18h] [rbp-48h]

  v3 = (__int64 **)*((_QWORD *)a1 + 1);
  v4 = *a1;
  v5 = *v3;
  if ( (_WORD)v4 )
  {
    if ( (unsigned __int16)(v4 - 17) > 0xD3u )
    {
      LOWORD(v22) = *a1;
      v23 = v3;
LABEL_4:
      if ( (_WORD)v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
        BUG();
      v6 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v4 - 16];
      if ( (_DWORD)v6 == 1 )
        goto LABEL_7;
      goto LABEL_16;
    }
    LOWORD(v4) = word_4456580[v4 - 1];
    v16 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)a1) )
    {
      v23 = v3;
      LOWORD(v22) = 0;
      goto LABEL_15;
    }
    LOWORD(v4) = sub_3009970((__int64)a1, a2, v18, v19, v20);
  }
  LOWORD(v22) = v4;
  v23 = v16;
  if ( (_WORD)v4 )
    goto LABEL_4;
LABEL_15:
  LODWORD(v6) = sub_3007260((__int64)&v22);
  if ( (_DWORD)v6 == 1 )
  {
LABEL_7:
    v7 = 2;
    goto LABEL_8;
  }
LABEL_16:
  switch ( (_DWORD)v6 )
  {
    case 2:
      v7 = 3;
      break;
    case 4:
      v7 = 4;
      break;
    case 8:
      v7 = 5;
      break;
    case 0x10:
      v7 = 6;
      break;
    case 0x20:
      v7 = 7;
      break;
    case 0x40:
      v7 = 8;
      break;
    case 0x80:
      v7 = 9;
      break;
    default:
      v7 = sub_3007020(v5, v6);
      LODWORD(v8) = *a1;
      v9 = v17;
      if ( (_WORD)v8 )
        goto LABEL_9;
      goto LABEL_24;
  }
LABEL_8:
  LODWORD(v8) = *a1;
  v9 = 0;
  if ( (_WORD)v8 )
  {
LABEL_9:
    v10 = (unsigned __int16)(v8 - 176) <= 0x34u;
    LODWORD(v11) = word_4456340[(int)v8 - 1];
    LOBYTE(v8) = v10;
    goto LABEL_10;
  }
LABEL_24:
  v11 = sub_3007240((__int64)a1);
  v8 = HIDWORD(v11);
  v22 = v11;
  v10 = BYTE4(v11);
LABEL_10:
  LODWORD(v21) = v11;
  BYTE4(v21) = v8;
  if ( !v10 )
  {
    LOWORD(v12) = sub_2D43050(v7, v11);
    if ( (_WORD)v12 )
      goto LABEL_12;
LABEL_28:
    v12 = sub_3009450(v5, v7, v9, v21, v13, v14);
    HIWORD(v2) = HIWORD(v12);
    goto LABEL_12;
  }
  LOWORD(v12) = sub_2D43AD0(v7, v11);
  if ( !(_WORD)v12 )
    goto LABEL_28;
LABEL_12:
  LOWORD(v2) = v12;
  return v2;
}
