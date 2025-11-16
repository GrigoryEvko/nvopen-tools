// Function: sub_350FA40
// Address: 0x350fa40
//
__int64 __fastcall sub_350FA40(unsigned __int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned __int64 v4; // r13
  char v6; // dl
  __int64 v7; // rdi
  bool v8; // al
  unsigned int v9; // eax
  __int64 v10; // rax
  int v11; // esi
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // r15
  int v18; // esi
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v23; // [rsp+0h] [rbp-40h] BYREF
  bool v24; // [rsp+8h] [rbp-38h]

  v4 = a1 >> 3;
  if ( (a1 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
  {
    if ( (a1 & 1) != 0 )
      goto LABEL_27;
    v11 = (unsigned __int16)((unsigned int)a1 >> 8);
    v8 = (a1 & 8) != 0;
    goto LABEL_19;
  }
  v6 = a1 & 2;
  if ( (a1 & 4) == 0 )
  {
    if ( (a1 & 6) == 2 || (a1 & 1) != 0 )
    {
      if ( v6 )
      {
        v7 = HIWORD(a1);
LABEL_7:
        v8 = 0;
LABEL_8:
        v23 = v7;
        v24 = v8;
        v9 = sub_CA1930(&v23);
        switch ( v9 )
        {
          case 1u:
            LOWORD(v10) = 2;
            break;
          case 2u:
            LOWORD(v10) = 3;
            break;
          case 4u:
            LOWORD(v10) = 4;
            break;
          case 8u:
            LOWORD(v10) = 5;
            break;
          case 0x10u:
            LOWORD(v10) = 6;
            break;
          case 0x20u:
            LOWORD(v10) = 7;
            break;
          case 0x40u:
            LOWORD(v10) = 8;
            break;
          case 0x80u:
            LOWORD(v10) = 9;
            break;
          default:
            v10 = sub_3007020(a2, v9);
            v2 = v10;
            break;
        }
        LOWORD(v2) = v10;
        return v2;
      }
LABEL_27:
      v7 = HIDWORD(a1);
      goto LABEL_7;
    }
    v8 = (a1 & 8) != 0;
    v11 = (unsigned __int16)((unsigned int)a1 >> 8);
    if ( v6 )
    {
      v7 = v11 * (unsigned int)HIWORD(a1);
      goto LABEL_8;
    }
LABEL_19:
    v7 = (unsigned int)(v11 * HIDWORD(a1));
    goto LABEL_8;
  }
  if ( v6 )
  {
    v12 = v4 & 0xFFFFFFFFFFE00000LL;
    v13 = 1;
    v14 = 0;
  }
  else
  {
    v12 = v4 & 0xFFFFFFFFE0000000LL;
    v13 = 0;
    v14 = 1;
  }
  v15 = sub_350FA40((8 * v12) | v14 | (2 * v13), a2);
  v17 = v16;
  v18 = (unsigned __int16)(v4 >> 5);
  LODWORD(v23) = v18;
  BYTE4(v23) = v4 & 1;
  if ( (v4 & 1) != 0 )
    LOWORD(v19) = sub_2D43AD0(v15, v18);
  else
    LOWORD(v19) = sub_2D43050(v15, v18);
  if ( !(_WORD)v19 )
  {
    v19 = sub_3009450(a2, v15, v17, v23, v20, v21);
    v3 = v19;
  }
  LOWORD(v3) = v19;
  return v3;
}
