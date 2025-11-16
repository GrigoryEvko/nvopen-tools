// Function: sub_15334D0
// Address: 0x15334d0
//
void __fastcall sub_15334D0(_QWORD **a1, __int64 *a2, __int64 a3, __int64 a4, unsigned int **a5, __int64 a6)
{
  __int64 *v7; // r14
  __int64 v10; // r10
  _BYTE *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // [rsp-58h] [rbp-58h]
  __int64 *v14; // [rsp-50h] [rbp-50h]
  unsigned int v15; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v16; // [rsp-44h] [rbp-44h] BYREF
  _QWORD v17[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( a3 )
  {
    v7 = a2;
    v15 = 0;
    v16 = 0;
    v14 = &a2[a3];
    if ( a2 != v14 )
    {
      do
      {
        v10 = *v7;
        if ( a6 )
        {
          v11 = *(_BYTE **)(a6 + 8);
          v12 = *((unsigned int *)*a1 + 2) + 8LL * *(unsigned int *)(**a1 + 8LL);
          v17[0] = v12;
          if ( v11 == *(_BYTE **)(a6 + 16) )
          {
            v13 = v10;
            sub_A235E0(a6, v11, v17);
            v10 = v13;
          }
          else
          {
            if ( v11 )
            {
              *(_QWORD *)v11 = v12;
              v11 = *(_BYTE **)(a6 + 8);
            }
            *(_QWORD *)(a6 + 8) = v11 + 8;
          }
        }
        if ( (unsigned __int8)(*(_BYTE *)v10 - 4) > 0x1Eu )
        {
          v17[0] = (unsigned int)sub_1524C80((__int64)(a1 + 3), **(_QWORD **)(v10 + 136));
          sub_1525CA0(a4, v17);
          v17[0] = (unsigned int)sub_153E840(a1 + 3);
          sub_1525CA0(a4, v17);
          sub_152B6B0(*a1, 2u, a4, 0);
          *(_DWORD *)(a4 + 8) = 0;
        }
        else
        {
          switch ( *(_BYTE *)v10 )
          {
            case 5:
              if ( a5 )
                sub_152F000(a1, v10, a4, *a5 + 1);
              else
                sub_152F000(a1, v10, a4, &v15);
              break;
            case 6:
              if ( a5 )
                sub_152BA30((_DWORD **)a1, v10, a4, (*a5)[2]);
              else
                sub_152BA30((_DWORD **)a1, v10, a4, 0);
              break;
            case 7:
              if ( a5 )
                sub_152BB80((_DWORD **)a1, v10, a4, (*a5)[3]);
              else
                sub_152BB80((_DWORD **)a1, v10, a4, 0);
              break;
            case 8:
              if ( a5 )
                sub_152F1E0((_DWORD **)a1, v10, a4, *a5 + 4);
              else
                sub_152F1E0((_DWORD **)a1, v10, a4, &v16);
              break;
            case 9:
              if ( a5 )
                sub_152BCD0((_DWORD **)a1, v10, a4, (*a5)[5]);
              else
                sub_152BCD0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0xA:
              if ( a5 )
                sub_152BE00((_DWORD **)a1, v10, a4, (*a5)[6]);
              else
                sub_152BE00((_DWORD **)a1, v10, a4, 0);
              break;
            case 0xB:
              if ( a5 )
                sub_152BF40((_DWORD **)a1, v10, a4, (*a5)[7]);
              else
                sub_152BF40((_DWORD **)a1, v10, a4, 0);
              break;
            case 0xC:
              if ( a5 )
                sub_152C120((_DWORD **)a1, v10, a4, (*a5)[8]);
              else
                sub_152C120((_DWORD **)a1, v10, a4, 0);
              break;
            case 0xD:
              if ( a5 )
                sub_152C340((_DWORD **)a1, v10, a4, (*a5)[9]);
              else
                sub_152C340((_DWORD **)a1, v10, a4, 0);
              break;
            case 0xE:
              if ( a5 )
                sub_152C5C0((_DWORD **)a1, v10, a4, (*a5)[10]);
              else
                sub_152C5C0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0xF:
              if ( a5 )
                sub_152C730((_DWORD **)a1, v10, a4, (*a5)[11]);
              else
                sub_152C730((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x10:
              if ( a5 )
                sub_152C920((_DWORD **)a1, v10, a4, (*a5)[12]);
              else
                sub_152C920((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x11:
              if ( a5 )
                sub_152CBC0((_DWORD **)a1, v10, a4, (*a5)[13]);
              else
                sub_152CBC0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x12:
              if ( a5 )
                sub_152CEF0((_DWORD **)a1, v10, a4, (*a5)[14]);
              else
                sub_152CEF0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x13:
              if ( a5 )
                sub_152D0D0((_DWORD **)a1, v10, a4, (*a5)[15]);
              else
                sub_152D0D0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x14:
              if ( a5 )
                sub_152D270((_DWORD **)a1, v10, a4, (*a5)[16]);
              else
                sub_152D270((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x15:
              if ( a5 )
                sub_152D3D0((__int64)a1, v10, a4, (*a5)[17]);
              else
                sub_152D3D0((__int64)a1, v10, a4, 0);
              break;
            case 0x16:
              if ( a5 )
                sub_152D540((_DWORD **)a1, v10, a4, (*a5)[18]);
              else
                sub_152D540((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x17:
              if ( a5 )
                sub_152D690((_DWORD **)a1, v10, a4, (*a5)[19]);
              else
                sub_152D690((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x18:
              if ( a5 )
                sub_152D870((_DWORD **)a1, v10, a4, (*a5)[20]);
              else
                sub_152D870((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x19:
              if ( a5 )
                sub_152DA80((_DWORD **)a1, v10, a4, (*a5)[21]);
              else
                sub_152DA80((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x1A:
              if ( a5 )
                sub_152DCE0((_DWORD **)a1, v10, a4, (*a5)[22]);
              else
                sub_152DCE0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x1B:
              if ( a5 )
                sub_152DEC0((_DWORD **)a1, v10, a4, (*a5)[23]);
              else
                sub_152DEC0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x1C:
              if ( a5 )
                sub_152E0F0((_DWORD **)a1, v10, a4, (*a5)[24]);
              else
                sub_152E0F0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x1D:
              if ( a5 )
                sub_152E350((_DWORD **)a1, v10, a4, (*a5)[25]);
              else
                sub_152E350((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x1E:
              if ( a5 )
                sub_152E520((_DWORD **)a1, v10, a4, (*a5)[26]);
              else
                sub_152E520((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x1F:
              if ( a5 )
                sub_152E6F0((_DWORD **)a1, v10, a4, (*a5)[27]);
              else
                sub_152E6F0((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x20:
              if ( a5 )
                sub_152E930((_DWORD **)a1, v10, a4, (*a5)[28]);
              else
                sub_152E930((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x21:
              if ( a5 )
                sub_152EB90((_DWORD **)a1, v10, a4, (*a5)[29]);
              else
                sub_152EB90((_DWORD **)a1, v10, a4, 0);
              break;
            case 0x22:
              if ( a5 )
                sub_152EDC0((_DWORD **)a1, v10, a4, (*a5)[30]);
              else
                sub_152EDC0((_DWORD **)a1, v10, a4, 0);
              break;
            default:
              if ( a5 )
                sub_152B8F0((__int64)a1, v10, a4, **a5);
              else
                sub_152B8F0((__int64)a1, v10, a4, 0);
              break;
          }
        }
        ++v7;
      }
      while ( v14 != v7 );
    }
  }
}
