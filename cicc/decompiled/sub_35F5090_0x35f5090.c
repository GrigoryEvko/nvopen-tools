// Function: sub_35F5090
// Address: 0x35f5090
//
void __fastcall sub_35F5090(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  size_t v8; // rax
  void *v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  size_t v12; // rdx
  char *v13; // rsi
  __int64 v14; // rdx
  unsigned int v15; // r13d
  void *v16; // rdx
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  unsigned int v19; // r13d
  void *v20; // rdx
  void *v21; // rdx
  __int64 v22; // rdx

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( a5 )
  {
    v8 = strlen((const char *)a5);
    if ( v8 == 3 )
    {
      if ( *(_WORD *)a5 == 29795 && *(_BYTE *)(a5 + 2) == 97 )
      {
        v9 = *(void **)(a4 + 32);
        v10 = *(_QWORD *)(a4 + 24) - (_QWORD)v9;
        if ( (v5 & 1) != 0 )
        {
          if ( v10 > 0xC )
          {
            qmemcpy(v9, ".cta_group::2", 13);
            *(_QWORD *)(a4 + 32) += 13LL;
            return;
          }
          v12 = 13;
          v13 = ".cta_group::2";
        }
        else
        {
          if ( v10 > 0xC )
          {
            qmemcpy(v9, ".cta_group::1", 13);
            *(_QWORD *)(a4 + 32) += 13LL;
            return;
          }
          v12 = 13;
          v13 = ".cta_group::1";
        }
        goto LABEL_17;
      }
      if ( *(_WORD *)a5 == 28006 && *(_BYTE *)(a5 + 2) == 116 && (v5 & 0x40) != 0 )
      {
        if ( (v5 & 0x100) != 0 )
          sub_C64ED0("Unsupported tcgen05.cp destination format", 1u);
        v14 = *(_QWORD *)(a4 + 32);
        v15 = ((unsigned int)v5 >> 7) & 1;
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v14) <= 5 )
        {
          sub_CB6200(a4, ".b8x16", 6u);
          v16 = *(void **)(a4 + 32);
        }
        else
        {
          *(_DWORD *)v14 = 2016961070;
          *(_WORD *)(v14 + 4) = 13873;
          v16 = (void *)(*(_QWORD *)(a4 + 32) + 6LL);
          *(_QWORD *)(a4 + 32) = v16;
        }
        v17 = *(_QWORD *)(a4 + 24) - (_QWORD)v16;
        if ( !(_BYTE)v15 )
        {
          if ( v17 > 9 )
          {
            qmemcpy(v16, ".b6x16_p32", 10);
            *(_QWORD *)(a4 + 32) += 10LL;
            return;
          }
          v12 = 10;
          v13 = ".b6x16_p32";
LABEL_17:
          sub_CB6200(a4, (unsigned __int8 *)v13, v12);
          return;
        }
        if ( v17 <= 9 )
        {
          v12 = 10;
          v13 = ".b4x16_p64";
          goto LABEL_17;
        }
        qmemcpy(v16, ".b4x16_p64", 10);
        *(_QWORD *)(a4 + 32) += 10LL;
      }
    }
    else if ( v8 == 5 )
    {
      if ( *(_DWORD *)a5 == 1885431923 && *(_BYTE *)(a5 + 4) == 101 )
      {
        v11 = *(_QWORD *)(a4 + 32);
        switch ( ((unsigned int)v5 >> 1) & 7 )
        {
          case 0u:
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 8 )
            {
              *(_BYTE *)(v11 + 8) = 98;
              *(_QWORD *)v11 = 0x363532783832312ELL;
              *(_QWORD *)(a4 + 32) += 9LL;
              return;
            }
            v12 = 9;
            v13 = ".128x256b";
            goto LABEL_17;
          case 1u:
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 6 )
            {
              *(_DWORD *)v11 = 846738478;
              *(_WORD *)(v11 + 4) = 13877;
              *(_BYTE *)(v11 + 6) = 98;
              *(_QWORD *)(a4 + 32) += 7LL;
              return;
            }
            v12 = 7;
            v13 = ".4x256b";
            goto LABEL_17;
          case 2u:
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 8 )
            {
              *(_BYTE *)(v11 + 8) = 98;
              *(_QWORD *)v11 = 0x383231783832312ELL;
              *(_QWORD *)(a4 + 32) += 9LL;
              return;
            }
            v12 = 9;
            v13 = ".128x128b";
            goto LABEL_17;
          case 3u:
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 7 )
            {
              *(_QWORD *)v11 = 0x623832317834362ELL;
              *(_QWORD *)(a4 + 32) += 8LL;
              return;
            }
            v12 = 8;
            v13 = ".64x128b";
            goto LABEL_17;
          case 4u:
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 7 )
            {
              *(_QWORD *)v11 = 0x623832317832332ELL;
              *(_QWORD *)(a4 + 32) += 8LL;
              return;
            }
            v12 = 8;
            v13 = ".32x128b";
            break;
          default:
            sub_C64ED0("Unexpected tcgen05.cp shape", 1u);
        }
        goto LABEL_17;
      }
    }
    else if ( v8 == 9 && *(_QWORD *)a5 == 0x73616369746C756DLL && *(_BYTE *)(a5 + 8) == 116 )
    {
      v18 = ((unsigned int)v5 >> 4) & 3;
      if ( v18 )
      {
        v19 = ((unsigned int)v5 >> 1) & 7;
        if ( v18 == 1 && v19 == 3 )
        {
          v20 = *(void **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v20 <= 0xDu )
          {
            v12 = 14;
            v13 = ".warpx2::02_13";
            goto LABEL_17;
          }
          qmemcpy(v20, ".warpx2::02_13", 14);
          *(_QWORD *)(a4 + 32) += 14LL;
        }
        else if ( v18 == 2 && v19 == 3 )
        {
          v21 = *(void **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v21 <= 0xDu )
          {
            v12 = 14;
            v13 = ".warpx2::01_23";
            goto LABEL_17;
          }
          qmemcpy(v21, ".warpx2::01_23", 14);
          *(_QWORD *)(a4 + 32) += 14LL;
        }
        else
        {
          if ( v18 != 3 || v19 != 4 )
            sub_C64ED0("Unsupported tcgen05.cp shape and multicast flags", 1u);
          v22 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v22) <= 6 )
          {
            v12 = 7;
            v13 = ".warpx4";
            goto LABEL_17;
          }
          *(_DWORD *)v22 = 1918990126;
          *(_WORD *)(v22 + 4) = 30832;
          *(_BYTE *)(v22 + 6) = 52;
          *(_QWORD *)(a4 + 32) += 7LL;
        }
      }
    }
  }
}
