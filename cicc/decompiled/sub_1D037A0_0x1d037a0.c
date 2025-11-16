// Function: sub_1D037A0
// Address: 0x1d037a0
//
__int64 __fastcall sub_1D037A0(__int64 **a1)
{
  __int64 *v1; // r8
  __int64 *v2; // r14
  __int64 *v3; // rbx
  __int64 v4; // r12
  bool v5; // al
  __int64 v6; // r13
  unsigned __int8 v7; // dl
  unsigned __int8 v8; // al
  char v10; // al
  int v11; // eax
  __int64 *v12; // [rsp+0h] [rbp-40h]
  __int64 *v13; // [rsp+8h] [rbp-38h]
  char v14; // [rsp+8h] [rbp-38h]
  __int64 *v15; // [rsp+8h] [rbp-38h]

  v1 = a1[3];
  v2 = a1[2];
  if ( v1 == v2 )
    return 0;
  v3 = v2 + 1;
  v4 = *v2;
  if ( v1 != v2 + 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = *v3;
        v7 = (*(_BYTE *)(v4 + 229) & 0x10) != 0;
        v8 = (*(_BYTE *)(*v3 + 229) & 0x10) != 0;
        if ( v7 == v8 )
          break;
        if ( v7 < v8 )
        {
          v4 = *v3;
          v2 = v3;
        }
        if ( v1 == ++v3 )
        {
LABEL_14:
          v3 = a1[3];
          goto LABEL_15;
        }
      }
      if ( (*(_BYTE *)(v4 + 228) & 2) != 0 || (*(_BYTE *)(v6 + 228) & 2) != 0 )
        break;
      v12 = v1;
      v14 = sub_1D01A90(a1[21], v4);
      v10 = sub_1D01A90(a1[21], v6);
      v1 = v12;
      if ( v10 != 1 && v14 )
      {
LABEL_21:
        v4 = *v3;
        v2 = v3;
        goto LABEL_9;
      }
      if ( v14 == 1 )
        goto LABEL_26;
      if ( v10 )
        goto LABEL_8;
      if ( v14 || (v11 = sub_1D02CA0(v4, v6, 1, (__int64)a1[21]), v1 = v12, !v11) )
      {
LABEL_26:
        v15 = v1;
        v5 = sub_1D03130(v4, v6, (__int64)a1[21]);
        v1 = v15;
LABEL_7:
        if ( v5 )
          goto LABEL_21;
        goto LABEL_8;
      }
      if ( v11 > 0 )
        goto LABEL_21;
LABEL_8:
      v4 = *v2;
LABEL_9:
      if ( v1 == ++v3 )
        goto LABEL_14;
    }
    v13 = v1;
    v5 = sub_1D03130(v4, *v3, (__int64)a1[21]);
    v1 = v13;
    goto LABEL_7;
  }
LABEL_15:
  if ( v2 != v3 - 1 )
  {
    *v2 = *(v3 - 1);
    *(v3 - 1) = v4;
    v2 = a1[3] - 1;
  }
  a1[3] = v2;
  *(_DWORD *)(v4 + 196) = 0;
  return v4;
}
