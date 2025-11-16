// Function: sub_1703ED0
// Address: 0x1703ed0
//
__int64 __fastcall sub_1703ED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  _QWORD *v7; // r12
  int v8; // r14d
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r8
  unsigned int v12; // edi
  _QWORD *v13; // rsi
  _QWORD *v14; // r10
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 **v17; // rax
  int v18; // eax
  int v19; // esi
  int v20; // ecx
  unsigned int v21; // r13d
  unsigned int v22; // eax
  unsigned int v23; // r12d
  unsigned int v25; // r13d
  _QWORD *v26; // rax
  __int64 *v27; // [rsp+8h] [rbp-48h]
  unsigned __int8 v28; // [rsp+17h] [rbp-39h]
  __int64 *v29; // [rsp+18h] [rbp-38h]

  if ( !(unsigned __int8)sub_1703010(a1, a2, a3, a4, a5, a6) )
    return 0;
  v7 = *(_QWORD **)(a1 + 72);
  v27 = *(__int64 **)(a1 + 120);
  if ( *(__int64 **)(a1 + 112) == v27 )
  {
    v25 = sub_16431D0(*(_QWORD *)*(v7 - 3));
    v23 = sub_1703900(a1);
    if ( v25 <= v23 )
      return 0;
    goto LABEL_30;
  }
  v29 = *(__int64 **)(a1 + 112);
  v8 = 0;
  do
  {
    v9 = *v29;
    v10 = *(_QWORD *)(*v29 + 8);
    if ( v10 && *(_QWORD *)(v10 + 8) )
    {
      v28 = *(_BYTE *)(v9 + 16) - 61;
      do
      {
        v15 = sub_1648700(v10);
        if ( *((_BYTE *)v15 + 16) <= 0x17u || v15 == v7 )
          goto LABEL_10;
        v16 = *(unsigned int *)(a1 + 104);
        if ( (_DWORD)v16 )
        {
          v11 = *(_QWORD *)(a1 + 88);
          v12 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v13 = (_QWORD *)(v11 + 16LL * v12);
          v14 = (_QWORD *)*v13;
          if ( v15 == (_QWORD *)*v13 )
          {
LABEL_9:
            if ( v13 != (_QWORD *)(v11 + 16 * v16) )
              goto LABEL_10;
          }
          else
          {
            v19 = 1;
            while ( v14 != (_QWORD *)-8LL )
            {
              v20 = v19 + 1;
              v12 = (v16 - 1) & (v19 + v12);
              v13 = (_QWORD *)(v11 + 16LL * v12);
              v14 = (_QWORD *)*v13;
              if ( v15 == (_QWORD *)*v13 )
                goto LABEL_9;
              v19 = v20;
            }
          }
        }
        if ( v28 > 1u )
          return 0;
        v17 = (*(_BYTE *)(v9 + 23) & 0x40) != 0
            ? *(__int64 ***)(v9 - 8)
            : (__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
        v18 = sub_16431D0(**v17);
        if ( v8 )
        {
          if ( v18 != v8 )
            return 0;
        }
        v8 = v18;
LABEL_10:
        v10 = *(_QWORD *)(v10 + 8);
      }
      while ( v10 );
    }
    v29 += 3;
  }
  while ( v27 != v29 );
  v21 = sub_16431D0(*(_QWORD *)*(v7 - 3));
  v22 = sub_1703900(a1);
  v23 = v22;
  if ( v21 <= v22 || v8 && v8 != v22 )
    return 0;
LABEL_30:
  v26 = (_QWORD *)sub_16498A0(*(_QWORD *)(a1 + 72));
  return sub_1644900(v26, v23);
}
