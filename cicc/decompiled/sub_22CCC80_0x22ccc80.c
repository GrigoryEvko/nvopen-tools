// Function: sub_22CCC80
// Address: 0x22ccc80
//
__int64 __fastcall sub_22CCC80(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int *v11; // rsi
  _BYTE *v12; // rdi
  __int64 v13; // rdx
  __int64 v15; // rax
  int v16; // eax
  _QWORD v17[8]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v8 = *(_QWORD *)(a3 - 32);
  if ( *(_BYTE *)v8 != 85 )
    goto LABEL_2;
  v15 = *(_QWORD *)(v8 - 32);
  if ( !v15 || *(_BYTE *)v15 || *(_QWORD *)(v15 + 24) != *(_QWORD *)(v8 + 80) || (*(_BYTE *)(v15 + 33) & 0x20) == 0 )
    goto LABEL_2;
  v16 = *(_DWORD *)(v15 + 36);
  if ( v16 != 312 )
  {
    switch ( v16 )
    {
      case 333:
      case 339:
      case 360:
      case 369:
      case 372:
        break;
      default:
        goto LABEL_2;
    }
  }
  if ( *(_DWORD *)(a3 + 80) == 1 && !**(_DWORD **)(a3 + 72) )
  {
    sub_22CB5B0(a1, a2, v8, a4);
  }
  else
  {
LABEL_2:
    v9 = sub_B43CC0(a3);
    v10 = *(unsigned int *)(a3 + 80);
    v11 = *(unsigned int **)(a3 + 72);
    v12 = *(_BYTE **)(a3 - 32);
    v18 = 257;
    v17[0] = v9;
    memset(&v17[1], 0, 56);
    v13 = sub_1002A30(v12, v11, v10);
    if ( v13 )
    {
      sub_22C7100(a1, a2, v13, a4, a3);
    }
    else
    {
      *(_BYTE *)(a1 + 40) = 1;
      *(_WORD *)a1 = 6;
      LOWORD(v17[0]) = 0;
      sub_22C0090((unsigned __int8 *)v17);
    }
  }
  return a1;
}
