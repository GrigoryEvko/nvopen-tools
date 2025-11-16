// Function: sub_30097B0
// Address: 0x30097b0
//
__int64 __fastcall sub_30097B0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // eax
  unsigned int v6; // r13d
  unsigned int v7; // r14d
  unsigned __int8 v8; // bl
  int v10; // r13d
  unsigned int v11; // eax
  __int64 *v12; // r12
  __int64 v13; // rdx
  unsigned int v14; // r15d
  int v15; // eax
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // esi
  __int16 v19; // cx
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v8 = *(_BYTE *)(a1 + 8);
  if ( v8 == 12 )
  {
    v18 = *(_DWORD *)(a1 + 8) >> 8;
    switch ( v18 )
    {
      case 1u:
        v19 = 2;
        break;
      case 2u:
        v19 = 3;
        break;
      case 4u:
        v19 = 4;
        break;
      case 8u:
        v19 = 5;
        break;
      case 0x10u:
        v19 = 6;
        break;
      case 0x20u:
        v19 = 7;
        break;
      case 0x40u:
        v19 = 8;
        break;
      case 0x80u:
        v19 = 9;
        break;
      default:
        v5 = sub_3007020(*(_QWORD **)a1, v18);
        v19 = v5;
        break;
    }
    LOWORD(v5) = v19;
    return v5;
  }
  else
  {
    if ( v8 <= 0xCu )
    {
      v6 = 264;
      if ( v8 == 11 )
        return v6;
LABEL_11:
      LOWORD(v6) = sub_3009540(a1, a2);
      return v6;
    }
    if ( (unsigned __int8)(v8 - 17) > 1u )
      goto LABEL_11;
    v10 = *(_DWORD *)(a1 + 32);
    v11 = sub_30097B0(*(_QWORD *)(a1 + 24), 0, a3, a4, a5);
    LODWORD(v21) = v10;
    v12 = *(__int64 **)a1;
    v20 = v13;
    v14 = v11;
    if ( v8 == 18 )
      LOWORD(v15) = sub_2D43AD0(v11, v10);
    else
      LOWORD(v15) = sub_2D43050(v11, v10);
    if ( !(_WORD)v15 )
    {
      BYTE4(v21) = v8 == 18;
      v15 = sub_3009450(v12, v14, v20, v21, v16, v17);
      HIWORD(v7) = HIWORD(v15);
    }
    LOWORD(v7) = v15;
    return v7;
  }
}
