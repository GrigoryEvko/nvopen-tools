// Function: sub_ED6100
// Address: 0xed6100
//
__int64 __fastcall sub_ED6100(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  __int64 v6; // rax
  _WORD *v8; // rdx
  _QWORD *v9; // rax
  int v10; // r11d
  int v11; // edi
  __int64 v12; // r9
  _QWORD *v13; // rdx
  char *v14; // rax
  __int64 v15; // rsi
  int v16; // ecx
  int v17; // edx

  v4 = *(_QWORD **)(a2 + 16);
  v6 = *(_QWORD *)(v4[2] + 8 * (a3 & (*v4 - 1LL)));
  if ( v6 && (v8 = (_WORD *)(v4[3] + v6), v9 = v8 + 1, v10 = (unsigned __int16)*v8, *v8) )
  {
    v11 = 0;
    while ( 1 )
    {
      v12 = v9[1];
      v13 = v9 + 3;
      if ( a3 == *v9 && a3 == v9[3] )
        break;
      ++v11;
      v9 = (_QWORD *)((char *)v13 + v12 + v9[2]);
      if ( v10 == v11 )
        goto LABEL_2;
    }
    v14 = (char *)v13 + v12;
    v15 = *(_QWORD *)((char *)v13 + v12);
    v16 = *(_DWORD *)((char *)v13 + v12 + 8);
    v17 = *(_DWORD *)((char *)v13 + v12 + 12);
    LOBYTE(v14) = v14[16];
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v15;
    *(_BYTE *)(a1 + 24) = (_BYTE)v14;
    *(_DWORD *)(a1 + 16) = v16;
    *(_DWORD *)(a1 + 20) = v17;
    return a1;
  }
  else
  {
LABEL_2:
    *(_QWORD *)a2 = a3;
    *(_BYTE *)(a2 + 8) = 1;
    *(_OWORD *)a1 = 0;
    *(_OWORD *)(a1 + 16) = 0;
    return a1;
  }
}
