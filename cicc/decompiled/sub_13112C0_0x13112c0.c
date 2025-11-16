// Function: sub_13112C0
// Address: 0x13112c0
//
void __fastcall sub_13112C0(__int64 a1)
{
  __int64 v2; // r13
  __int64 *v3; // r15
  __int64 v4; // r12
  unsigned __int16 v5; // dx
  int v6; // r8d
  __int16 v7; // di
  unsigned __int16 v8; // ax
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // rdx
  int v11; // r9d
  int v12; // eax
  unsigned __int8 v13; // al
  int v14; // r8d
  unsigned __int64 v15; // rax
  char v16; // dl
  __int64 *v17; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)a1 )
  {
    v2 = *(unsigned int *)(a1 + 304);
    v17 = (__int64 *)(a1 + 856);
    v3 = (__int64 *)(a1 + 24 * v2 + 864);
    v4 = a1 + 24 * v2;
    sub_1310140(a1, (__int64 *)(a1 + 856), v3, *(_DWORD *)(a1 + 304), (unsigned int)v2 <= 0x23);
    v5 = (unsigned __int16)(*(_WORD *)(v4 + 884) - *(_WORD *)(v4 + 880)) >> 3;
    if ( v5 )
    {
      v6 = (unsigned __int16)(*(_WORD *)(v4 + 884) - *(_WORD *)(v4 + 880)) >> 5;
      v7 = *(_QWORD *)(v4 + 864);
      v8 = (unsigned __int16)(*(_WORD *)(v4 + 884) - v7) >> 3;
      if ( (unsigned int)v2 > 0x23 )
      {
        sub_1310E90(a1, v17, (char **)v3, v2, v8 - v5 + v6);
        v7 = *(_WORD *)(v4 + 864);
      }
      else
      {
        v9 = v5 - v6;
        v10 = *(unsigned __int8 *)(a1 + v2 + 380);
        v11 = v9;
        if ( v9 >= v10 )
        {
          v14 = v8;
          v15 = qword_4F96AD0 / qword_505FA40[v2];
          if ( v15 > 0xFF )
            LOBYTE(v15) = -1;
          *(_BYTE *)(a1 + v2 + 380) = v15;
          sub_13108D0(a1, v17, v3, v2, v14 - v11);
          v16 = *(_BYTE *)(a1 + v2 + 308);
          if ( (int)*(unsigned __int16 *)(unk_5060A20 + 2 * v2) >> (v16 + 1) )
            *(_BYTE *)(a1 + v2 + 308) = v16 + 1;
          v7 = *(_WORD *)(v4 + 864);
        }
        else
        {
          *(_BYTE *)(a1 + v2 + 380) = v10 - v9;
        }
      }
    }
    else
    {
      v7 = *(_WORD *)(v4 + 864);
      if ( (unsigned int)v2 <= 0x23 && *(_BYTE *)(a1 + v2 + 344) )
      {
        v13 = *(_BYTE *)(a1 + v2 + 308);
        if ( v13 > 1u )
          *(_BYTE *)(a1 + v2 + 308) = v13 - 1;
        *(_BYTE *)(a1 + v2 + 344) = 0;
      }
    }
    *(_WORD *)(a1 + 24 * v2 + 880) = v7;
    v12 = *(_DWORD *)(a1 + 304) + 1;
    *(_DWORD *)(a1 + 304) = v12;
    if ( v12 == dword_5060A18[0] )
      *(_DWORD *)(a1 + 304) = 0;
  }
}
