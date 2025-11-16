// Function: sub_2576250
// Address: 0x2576250
//
char __fastcall sub_2576250(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 v4; // r8
  __int64 v5; // rsi
  int v6; // ecx
  int v7; // r12d
  __int64 v8; // rdi
  unsigned int i; // eax
  _QWORD *v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 v12; // rax
  int v13; // edx
  char v14; // dl

  v2 = *(_DWORD *)(a2 + 56);
  v4 = *(_QWORD *)(a2 + 40);
  if ( v2 )
  {
    v5 = *(_QWORD *)(a1 + 72);
    v6 = v2 - 1;
    v7 = 1;
    v8 = *(_QWORD *)(a1 + 80);
    for ( i = v6
            & (((unsigned int)v8 >> 9)
             ^ ((unsigned int)v8 >> 4)
             ^ (16 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = v6 & v11 )
    {
      v10 = (_QWORD *)(v4 + ((unsigned __int64)i << 6));
      if ( v5 == *v10 && v8 == v10[1] )
        break;
      if ( unk_4FEE4D0 == *v10 && qword_4FEE4D8 == v10[1] )
        goto LABEL_7;
      v11 = v7 + i;
      ++v7;
    }
    LOBYTE(v12) = *(_BYTE *)(a1 + 104);
    *(_BYTE *)(a1 + 105) = v12;
  }
  else
  {
LABEL_7:
    LOBYTE(v12) = *(_BYTE *)(a1 + 104);
    if ( *(_BYTE *)(a1 + 105) != (_BYTE)v12 )
    {
      v12 = sub_250D070((_QWORD *)(a1 + 72));
      v13 = *(unsigned __int8 *)v12;
      if ( (_BYTE)v13 == 17 )
      {
        v14 = *(_BYTE *)(a1 + 105);
        if ( v14 )
        {
          sub_2575FB0((_DWORD *)(a1 + 112), (const void **)(v12 + 24));
          LODWORD(v12) = *(_DWORD *)(a1 + 152);
          if ( (unsigned int)v12 < unk_4FEF868 )
          {
            v14 = *(_BYTE *)(a1 + 105);
            LOBYTE(v12) = (_DWORD)v12 == 0;
            *(_BYTE *)(a1 + 288) &= v12;
          }
          else
          {
            v14 = *(_BYTE *)(a1 + 104);
            *(_BYTE *)(a1 + 105) = v14;
          }
        }
        *(_BYTE *)(a1 + 104) = v14;
      }
      else if ( (unsigned int)(v13 - 12) <= 1 )
      {
        LOBYTE(v12) = *(_BYTE *)(a1 + 105);
        *(_BYTE *)(a1 + 288) = *(_DWORD *)(a1 + 152) == 0;
        *(_BYTE *)(a1 + 104) = v12;
      }
    }
  }
  return v12;
}
