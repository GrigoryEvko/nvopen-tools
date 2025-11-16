// Function: sub_BDBBF0
// Address: 0xbdbbf0
//
__int64 __fastcall sub_BDBBF0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned int *v12; // rax
  int v13; // ecx
  __int64 v14; // rsi
  __int64 v15; // rax

  if ( *(char *)(a2 + 7) >= 0 )
    goto LABEL_11;
  v5 = sub_BD2BC0(a2);
  v7 = v5 + v6;
  if ( *(char *)(a2 + 7) < 0 )
    v7 -= sub_BD2BC0(a2);
  v8 = v7 >> 4;
  if ( (_DWORD)v8 )
  {
    v9 = 0;
    v10 = 16LL * (unsigned int)v8;
    while ( 1 )
    {
      v11 = 0;
      if ( *(char *)(a2 + 7) < 0 )
        v11 = sub_BD2BC0(a2);
      v12 = (unsigned int *)(v9 + v11);
      if ( a3 == *(_DWORD *)(*(_QWORD *)v12 + 8LL) )
        break;
      v9 += 16;
      if ( v10 == v9 )
        goto LABEL_11;
    }
    v13 = *(_DWORD *)(a2 + 4);
    v14 = v12[2];
    *(_QWORD *)(a1 + 16) = *(_QWORD *)v12;
    v15 = v12[3];
    *(_BYTE *)(a1 + 24) = 1;
    v14 *= 32;
    *(_QWORD *)a1 = v14 - 32LL * (v13 & 0x7FFFFFF) + a2;
    *(_QWORD *)(a1 + 8) = (32 * v15 - v14) >> 5;
  }
  else
  {
LABEL_11:
    *(_BYTE *)(a1 + 24) = 0;
  }
  return a1;
}
