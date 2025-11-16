// Function: sub_1528330
// Address: 0x1528330
//
void __fastcall sub_1528330(_DWORD *a1, unsigned int a2, char *a3, int a4, unsigned int a5)
{
  _BYTE **v7; // r11
  char *v9; // r12
  unsigned int v10; // esi
  __int64 v11; // rax
  char *v12; // r15
  char v13; // dl
  int v14; // ebx
  unsigned int v15; // [rsp+Ch] [rbp-154h]
  _DWORD *v16; // [rsp+10h] [rbp-150h]
  _BYTE **v17; // [rsp+18h] [rbp-148h]
  _BYTE *v18; // [rsp+20h] [rbp-140h] BYREF
  __int64 v19; // [rsp+28h] [rbp-138h]
  _BYTE v20[304]; // [rsp+30h] [rbp-130h] BYREF

  v7 = &v18;
  v18 = v20;
  v19 = 0x4000000000LL;
  if ( a4 )
  {
    v9 = a3;
    v10 = 64;
    v11 = 0;
    v12 = &a3[a4 - 1];
    v7 = &v18;
    while ( 1 )
    {
      v13 = *v9;
      if ( a5 && (unsigned __int8)((v13 & 0xDF) - 65) > 0x19u )
      {
        if ( (unsigned __int8)(v13 - 46) > 0x31u )
        {
          a5 = 0;
          v14 = v13;
          if ( v10 > (unsigned int)v11 )
            goto LABEL_10;
          goto LABEL_15;
        }
        if ( ((0x2000000000FFDuLL >> (v13 - 46)) & 1) == 0 )
          a5 = 0;
      }
      v14 = v13;
      if ( v10 > (unsigned int)v11 )
        goto LABEL_10;
LABEL_15:
      v15 = a2;
      v16 = a1;
      v17 = v7;
      sub_16CD150(v7, v20, 0, 4);
      v11 = (unsigned int)v19;
      a2 = v15;
      a1 = v16;
      v7 = v17;
LABEL_10:
      *(_DWORD *)&v18[4 * v11] = v14;
      v11 = (unsigned int)(v19 + 1);
      LODWORD(v19) = v19 + 1;
      if ( v12 == v9 )
        break;
      v10 = HIDWORD(v19);
      ++v9;
    }
  }
  sub_1528260(a1, a2, (__int64)v7, a5);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
}
