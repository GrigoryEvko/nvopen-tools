// Function: sub_2B1FA70
// Address: 0x2b1fa70
//
__int64 __fastcall sub_2B1FA70(__int64 a1, unsigned int a2)
{
  __int64 v3; // r14
  unsigned int v4; // r12d
  int v5; // eax
  int v6; // esi
  __int64 *i; // rdi
  __int64 v8; // rax
  unsigned int v9; // r15d
  unsigned int v10; // eax
  unsigned int v11; // ecx
  unsigned int v12; // ecx
  __int64 v14; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(***(_QWORD ***)(a1 + 8) + 8LL);
  v4 = sub_2B1E190(*(_QWORD *)a1, v3, a2);
  v5 = *(unsigned __int8 *)(v3 + 8);
  if ( (_BYTE)v5 != 17 )
    goto LABEL_8;
LABEL_2:
  v6 = v4 * *(_DWORD *)(v3 + 32);
LABEL_3:
  for ( i = **(__int64 ***)(v3 + 16); ; i = (__int64 *)v3 )
  {
    v8 = sub_BCDA70(i, v6);
    v9 = sub_2B1F810(*(_QWORD *)a1, v8, 0xFFFFFFFF);
    v14 = *(_QWORD *)a1;
    sub_DFB180(*(__int64 **)a1, 1u);
    v10 = sub_DFB120(v14);
    if ( v9 <= v10 )
      break;
    if ( --v4 )
    {
      _BitScanReverse(&v11, v4);
      v4 = 0x80000000 >> (v11 ^ 0x1F);
    }
    v5 = *(unsigned __int8 *)(v3 + 8);
    if ( (_BYTE)v5 == 17 )
      goto LABEL_2;
LABEL_8:
    v6 = v4;
    if ( (unsigned int)(v5 - 17) <= 1 )
      goto LABEL_3;
  }
  if ( v10 >> 1 < v9 && v4 )
  {
    _BitScanReverse(&v12, v4);
    return 0x80000000 >> (v12 ^ 0x1F);
  }
  return v4;
}
