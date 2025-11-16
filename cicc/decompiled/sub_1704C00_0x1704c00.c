// Function: sub_1704C00
// Address: 0x1704c00
//
_QWORD *__fastcall sub_1704C00(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _QWORD *v9; // rbx
  int v10; // r8d
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rax
  int v14; // r8d
  __int64 *v15; // r11
  __int64 *v16; // rcx
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v21; // rax
  __int64 *v22; // rax
  int v23; // [rsp+0h] [rbp-50h]
  int v24; // [rsp+4h] [rbp-4Ch]
  unsigned int v27; // [rsp+18h] [rbp-38h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  v6 = a1;
  if ( !a1 )
  {
    v21 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v21 = **(_QWORD **)(v21 + 16);
    v6 = *(_QWORD *)(v21 + 24);
  }
  v27 = a4 + 1;
  v9 = sub_1648A60(72, (int)a4 + 1);
  if ( v9 )
  {
    v10 = v27;
    v11 = *(_QWORD *)a2;
    v28 = (__int64)&v9[-3 * v27];
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v11 = **(_QWORD **)(v11 + 16);
    v23 = v10;
    v24 = *(_DWORD *)(v11 + 8) >> 8;
    v12 = (__int64 *)sub_15F9F50(v6, (__int64)a3, a4);
    v13 = (__int64 *)sub_1646BA0(v12, v24);
    v14 = v23;
    v15 = v13;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v22 = sub_16463B0(v13, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
      v14 = v23;
      v15 = v22;
    }
    else
    {
      v16 = &a3[a4];
      if ( v16 != a3 )
      {
        v17 = a3;
        while ( 1 )
        {
          v18 = *(_QWORD *)*v17;
          if ( *(_BYTE *)(v18 + 8) == 16 )
            break;
          if ( v16 == ++v17 )
            goto LABEL_11;
        }
        v19 = sub_16463B0(v15, *(_QWORD *)(v18 + 32));
        v14 = v23;
        v15 = v19;
      }
    }
LABEL_11:
    sub_15F1EA0((__int64)v9, (__int64)v15, 32, v28, v14, a6);
    v9[7] = v6;
    v9[8] = sub_15F9F50(v6, (__int64)a3, a4);
    sub_15F9CE0((__int64)v9, a2, a3, a4, a5);
  }
  return v9;
}
