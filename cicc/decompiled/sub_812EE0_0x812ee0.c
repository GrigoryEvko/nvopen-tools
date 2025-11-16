// Function: sub_812EE0
// Address: 0x812ee0
//
void __fastcall sub_812EE0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rcx
  __int64 v3; // rdx
  __int64 v4; // rdx
  size_t v5; // rax
  char *v6; // rax
  char *v7; // rax
  __int64 v8; // rax
  char s[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = a2;
  if ( !*(_QWORD *)(a1 + 8) && ((*(_BYTE *)(a1 + 170) & 2) == 0 || *(_BYTE *)(a1 + 136) == 3) )
  {
    sprintf(s, "__V%lu", ++qword_4F18BB8);
    v5 = strlen(s);
    v6 = (char *)sub_7E1510(v5 + 1);
    v7 = strcpy(v6, s);
    *(_BYTE *)(a1 + 89) |= 8u;
    v2 = a2;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 24) = v7;
  }
  v3 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v3 = *(_QWORD *)(v3 + 96);
    if ( v3 )
    {
      if ( (*(_BYTE *)(a1 + 170) & 0x10) != 0 && **(_QWORD **)(a1 + 216) )
      {
        v4 = *(_QWORD *)(v3 + 32);
        switch ( *(_BYTE *)(v4 + 80) )
        {
          case 4:
          case 5:
            v8 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 80LL);
            break;
          case 6:
            v8 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v8 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v8 = *(_QWORD *)(v4 + 88);
            break;
          default:
            BUG();
        }
        v3 = *(_QWORD *)(v8 + 104);
      }
      else
      {
        v3 = 0;
      }
    }
  }
  sub_812C80(a1, 7u, v3, v2);
}
