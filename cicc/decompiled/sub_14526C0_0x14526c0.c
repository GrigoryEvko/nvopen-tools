// Function: sub_14526C0
// Address: 0x14526c0
//
__int64 __fastcall sub_14526C0(__int64 a1)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rax
  _BYTE *v4; // r12
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // rax
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // r14d
  unsigned int v15; // r13d
  __int64 *v16; // rax
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r14d
  unsigned int v22; // r13d
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // [rsp+0h] [rbp-50h]
  __int64 v31; // [rsp+0h] [rbp-50h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v35[56]; // [rsp+18h] [rbp-38h] BYREF

  switch ( *(_WORD *)(a1 + 24) )
  {
    case 0:
      return *(_QWORD *)(a1 + 32);
    case 1:
      v6 = sub_14526C0(*(_QWORD *)(a1 + 32));
      if ( !v6 )
        return 0;
      return sub_15A43B0(v6, *(_QWORD *)(a1 + 40), 0);
    case 2:
      v7 = sub_14526C0(*(_QWORD *)(a1 + 32));
      if ( !v7 )
        return 0;
      return sub_15A3CB0(v7, *(_QWORD *)(a1 + 40), 0);
    case 3:
      v8 = sub_14526C0(*(_QWORD *)(a1 + 32));
      if ( !v8 )
        return 0;
      return sub_15A4460(v8, *(_QWORD *)(a1 + 40), 0);
    case 4:
      v9 = (_QWORD *)sub_14526C0(**(_QWORD **)(a1 + 32));
      v4 = v9;
      if ( !v9 )
        return 0;
      v10 = *v9;
      if ( *(_BYTE *)(v10 + 8) == 15 )
      {
        v11 = *(_DWORD *)(v10 + 8);
        v12 = sub_16498A0(v4);
        v13 = sub_16471D0(v12, v11 >> 8);
        v4 = (_BYTE *)sub_15A4510(v4, v13, 0);
      }
      v14 = *(_QWORD *)(a1 + 40);
      if ( v14 == 1 )
        return (__int64)v4;
      v15 = 1;
      while ( 2 )
      {
        v16 = (__int64 *)sub_14526C0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * v15));
        v17 = (__int64)v16;
        if ( !v16 )
          return 0;
        v18 = *(_QWORD *)v4;
        v19 = *v16;
        if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 15 )
        {
          if ( *(_BYTE *)(v19 + 8) != 15 )
            goto LABEL_36;
        }
        else
        {
          if ( *(_BYTE *)(v19 + 8) != 15 )
          {
LABEL_24:
            v4 = (_BYTE *)sub_15A2B30(v4, v17, 0, 0);
            goto LABEL_25;
          }
          v32 = v17;
          v30 = *(_DWORD *)(v19 + 8) >> 8;
          v25 = sub_16498A0(v17);
          v26 = sub_16471D0(v25, v30);
          v27 = sub_15A4510(v32, v26, 0);
          if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) != 15 )
          {
            v18 = *(_QWORD *)v27;
            v17 = (__int64)v4;
            v4 = (_BYTE *)v27;
            if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 15 )
              goto LABEL_24;
LABEL_36:
            v24 = *(_QWORD *)(v18 + 24);
            if ( *(_BYTE *)(v24 + 8) == 13 )
            {
              v31 = v18;
              v33 = v17;
              v28 = sub_16498A0(v4);
              v29 = sub_1643350(v28);
              v17 = sub_15A4750(v33, v29, 1);
              v24 = *(_QWORD *)(v31 + 24);
            }
            v34 = v17;
            v35[4] = 0;
            v4 = (_BYTE *)sub_15A2E80(v24, (_DWORD)v4, (unsigned int)&v34, 1, 0, (unsigned int)v35, 0);
LABEL_25:
            if ( v14 == ++v15 )
              return (__int64)v4;
            continue;
          }
        }
        return 0;
      }
    case 5:
      v20 = sub_14526C0(**(_QWORD **)(a1 + 32));
      v4 = (_BYTE *)v20;
      if ( !v20 || *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 15 )
        return 0;
      v21 = *(_QWORD *)(a1 + 40);
      if ( v21 == 1 )
        return (__int64)v4;
      v22 = 1;
      while ( 1 )
      {
        v23 = sub_14526C0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * v22));
        if ( !v23 || *(_BYTE *)(*(_QWORD *)v23 + 8LL) == 15 )
          break;
        ++v22;
        v4 = (_BYTE *)sub_15A2C20(v4, v23, 0, 0);
        if ( v21 == v22 )
          return (__int64)v4;
      }
      return 0;
    case 6:
      v2 = (_QWORD *)sub_14526C0(*(_QWORD *)(a1 + 32));
      if ( !v2 )
        return 0;
      v3 = (_QWORD *)sub_14526C0(*(_QWORD *)(a1 + 40));
      if ( !v3 || *v3 != *v2 )
        return 0;
      return sub_15A2C70(v2, v3, 0);
    case 0xA:
      v4 = *(_BYTE **)(a1 - 8);
      if ( v4[16] > 0x10u )
        return 0;
      return (__int64)v4;
    default:
      return 0;
  }
}
