// Function: sub_2699D80
// Address: 0x2699d80
//
__int64 __fastcall sub_2699D80(__int64 a1, _QWORD *a2)
{
  unsigned __int8 *v2; // r14
  unsigned __int64 v3; // r12
  char v4; // al
  int v5; // eax
  _QWORD *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  unsigned __int64 v13; // [rsp-50h] [rbp-50h] BYREF
  _QWORD *v14; // [rsp-48h] [rbp-48h] BYREF
  unsigned __int64 *v15; // [rsp-40h] [rbp-40h]

  if ( !*(_BYTE *)(a1 + 112) )
    return 1;
  v2 = *(unsigned __int8 **)(a1 + 104);
  if ( !v2 )
    return 1;
  v3 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v3 = *(_QWORD *)(v3 + 24);
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 <= 0x1Cu )
  {
    if ( v4 == 22 )
    {
      if ( !sub_B2FC80(*(_QWORD *)(v3 + 24)) )
      {
        v11 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 80LL);
        if ( !v11 )
          BUG();
        goto LABEL_25;
      }
      v4 = *(_BYTE *)v3;
    }
    if ( v4 || sub_B2FC80(v3) )
      goto LABEL_31;
    v11 = *(_QWORD *)(v3 + 80);
    if ( !v11 )
      BUG();
LABEL_25:
    v12 = *(_QWORD *)(v11 + 32);
    if ( v12 )
    {
      v2 = *(unsigned __int8 **)(a1 + 104);
      v3 = v12 - 24;
      goto LABEL_6;
    }
LABEL_31:
    v14 = 0;
    v15 = 0;
    BUG();
  }
LABEL_6:
  v14 = 0;
  v15 = 0;
  v5 = *(unsigned __int8 *)v3;
  if ( !(_BYTE)v5
    || (unsigned __int8)v5 > 0x1Cu
    && (v9 = (unsigned int)(v5 - 34), (unsigned __int8)v9 <= 0x33u)
    && (v10 = 0x8000000000041LL, _bittest64(&v10, v9)) )
  {
    v6 = (_QWORD *)(v3 & 0xFFFFFFFFFFFFFFFCLL | 2);
  }
  else
  {
    v6 = (_QWORD *)(v3 & 0xFFFFFFFFFFFFFFFCLL);
  }
  v14 = v6;
  nullsub_1518();
  sub_256F570((__int64)a2, (__int64)v14, (__int64)v15, v2, 1u);
  sub_2570110((__int64)a2, v3);
  if ( (unsigned __int8)(*(_BYTE *)v3 - 34) > 0x33u )
    return 0;
  v7 = 0x8000000000041LL;
  if ( !_bittest64(&v7, (unsigned int)*(unsigned __int8 *)v3 - 34) )
    return 0;
  v13 = v3;
  v14 = (_QWORD *)a1;
  v15 = &v13;
  if ( !byte_4FF47A8 )
    return 0;
  sub_267C110(a2, v3, "OMP180", 6u, &v14);
  return 0;
}
