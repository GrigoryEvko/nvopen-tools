// Function: sub_3143F80
// Address: 0x3143f80
//
__int64 __fastcall sub_3143F80(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // rax
  _QWORD *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  float v12; // xmm0_4
  float v13; // xmm0_4
  __int64 v14; // rax
  unsigned __int8 v15; // dl
  __int64 *v16; // rax
  __int64 v17; // rax
  int v18; // eax

  v2 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 85 )
  {
    if ( v2 != 34 && v2 != 40 )
    {
LABEL_4:
      *(_BYTE *)(a1 + 20) = 0;
      return a1;
    }
LABEL_10:
    sub_3143F30(a1, a2);
    return a1;
  }
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 )
    goto LABEL_10;
  if ( *(_BYTE *)v4
    || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80)
    || (*(_BYTE *)(v4 + 33) & 0x20) == 0
    || *(_DWORD *)(v4 + 36) != 291 )
  {
    if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      goto LABEL_4;
    goto LABEL_10;
  }
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(_QWORD *)(a2 + 32 * (1 - v5));
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = *(_QWORD *)(a2 + 32 * (2 - v5));
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_QWORD *)(a2 + 32 * (3 - v5));
  v11 = *(_QWORD *)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = *(_QWORD *)v11;
  if ( v11 < 0 )
    v12 = (float)(v11 & 1 | (unsigned int)((unsigned __int64)v11 >> 1))
        + (float)(v11 & 1 | (unsigned int)((unsigned __int64)v11 >> 1));
  else
    v12 = (float)(int)v11;
  v13 = v12 * 5.4210109e-20;
  if ( *(_QWORD *)(a2 + 48)
    && ((v14 = sub_B10CD0(a2 + 48), v15 = *(_BYTE *)(v14 - 16), (v15 & 2) == 0)
      ? (v16 = (__int64 *)(v14 - 16 - 8LL * ((v15 >> 2) & 0xF)))
      : (v16 = *(__int64 **)(v14 - 32)),
        v17 = *v16,
        *(_BYTE *)v17 == 20) )
  {
    v18 = *(_DWORD *)(v17 + 4);
  }
  else
  {
    v18 = 0;
  }
  *(_DWORD *)a1 = (_DWORD)v7;
  *(_DWORD *)(a1 + 8) = (_DWORD)v9;
  *(_DWORD *)(a1 + 12) = v18;
  *(_DWORD *)(a1 + 4) = 0;
  *(_BYTE *)(a1 + 20) = 1;
  *(float *)(a1 + 16) = v13;
  return a1;
}
