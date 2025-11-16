// Function: sub_2A70810
// Address: 0x2a70810
//
void __fastcall sub_2A70810(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  unsigned int v3; // ecx
  unsigned __int8 *v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rax
  int v7; // eax
  unsigned __int8 v8[48]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v9[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 15
    || (v9[0] = a2, *(_BYTE *)sub_2A686D0(a1 + 136, v9) == 6)
    || *(_DWORD *)(a2 + 80) != 1
    || (v2 = *(_QWORD *)(a2 - 32), *(_BYTE *)(*(_QWORD *)(v2 + 8) + 8LL) != 15) )
  {
    sub_2A6A450(a1, a2);
  }
  else
  {
    v3 = **(_DWORD **)(a2 + 72);
    if ( *(_BYTE *)v2 == 85
      && (v6 = *(_QWORD *)(v2 - 32)) != 0
      && !*(_BYTE *)v6
      && *(_QWORD *)(v6 + 24) == *(_QWORD *)(v2 + 80)
      && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
    {
      v7 = *(_DWORD *)(v6 + 36);
      if ( v7 != 312 )
      {
        switch ( v7 )
        {
          case 333:
          case 339:
          case 360:
          case 369:
          case 372:
            break;
          default:
            goto LABEL_6;
        }
      }
      sub_2A70140(a1, a2, *(_QWORD *)(a2 - 32), v3);
    }
    else
    {
LABEL_6:
      v4 = sub_2A6A1C0(a1, *(unsigned __int8 **)(a2 - 32), v3);
      sub_22C05A0((__int64)v8, v4);
      sub_22C05A0((__int64)v9, v8);
      v5 = sub_2A68BC0(a1, (unsigned __int8 *)a2);
      sub_2A639B0(a1, v5, a2, (__int64)v9, 0x100000000LL);
      sub_22C0090((unsigned __int8 *)v9);
      sub_22C0090(v8);
    }
  }
}
