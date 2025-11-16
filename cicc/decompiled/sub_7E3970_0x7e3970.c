// Function: sub_7E3970
// Address: 0x7e3970
//
__int64 __fastcall sub_7E3970(__int64 a1, __int64 a2, _QWORD *a3, __int64 *a4, _WORD *a5)
{
  __int64 v6; // rax
  unsigned int v10; // r8d
  __int64 v12; // rbx
  __int64 v13; // rbx
  _WORD *v14; // r9
  int i; // r10d
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int16 v19; // ax
  bool v20; // zf
  __int64 v21; // rbx
  __int64 v22; // rax
  char v23; // cl
  _QWORD *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h]
  int v26; // [rsp+14h] [rbp-5Ch]
  _WORD *v27; // [rsp+18h] [rbp-58h]
  _WORD *v28; // [rsp+18h] [rbp-58h]
  _WORD *v29; // [rsp+28h] [rbp-48h]
  _WORD *v30; // [rsp+28h] [rbp-48h]
  _WORD v31[25]; // [rsp+3Eh] [rbp-32h] BYREF

  v6 = a1;
  v31[0] = 0;
  if ( a2 )
    v6 = *(_QWORD *)(a2 + 40);
  v10 = 0;
  if ( (*(_BYTE *)(v6 + 176) & 0x10) != 0 )
  {
    v12 = *(_QWORD *)(v6 + 168);
    v25 = v12;
    v29 = a5;
    sub_7E3660(a1, a2, 0, a3, a4, a5, v31);
    v13 = *(_QWORD *)(v12 + 16);
    if ( !v13 )
      return v31[0];
    v14 = v29;
    for ( i = 0; ; i = 1 )
    {
      while ( 2 )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(v13 + 40) + 168LL);
        if ( i )
        {
          if ( (*(_BYTE *)(v13 + 96) & 2) == 0 )
            goto LABEL_14;
          if ( a2 )
            goto LABEL_14;
          v17 = v13;
          if ( !*(_QWORD *)(v16 + 224) )
            goto LABEL_14;
        }
        else
        {
          if ( (*(_BYTE *)(v13 + 96) & 3) != 1 || !*(_QWORD *)(v16 + 224) )
            goto LABEL_14;
          v17 = v13;
          if ( a2 )
          {
            v27 = v14;
            v18 = sub_8E5650(v13);
            v14 = v27;
            i = 0;
            v17 = v18;
          }
        }
        v26 = i;
        v28 = v14;
        v19 = sub_7E3970(a1, v17, a3, a4, v14);
        v20 = v31[0] == 0;
        v14 = v28;
        i = v26;
        *(_WORD *)(v17 + 138) = v19;
        if ( v20 )
          v31[0] = v19;
LABEL_14:
        v13 = *(_QWORD *)(v13 + 16);
        if ( v13 )
          continue;
        break;
      }
      if ( i == 1 )
        return v31[0];
      v21 = *(_QWORD *)(v25 + 16);
      if ( !v21 )
        return v31[0];
      do
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)(v21 + 40);
          if ( (*(_WORD *)(v22 + 176) & 0x110) == 0 )
            goto LABEL_26;
          v23 = *(_BYTE *)(v21 + 96) & 2;
          if ( (*(_BYTE *)(v22 + 176) & 0x10) != 0 )
            break;
          if ( v23 )
            goto LABEL_30;
          if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v21 + 112) + 8LL) + 16LL) + 96LL) & 2) != 0 )
            goto LABEL_21;
LABEL_26:
          v21 = *(_QWORD *)(v21 + 16);
          if ( !v21 )
            goto LABEL_27;
        }
        if ( v23 )
          goto LABEL_30;
LABEL_21:
        if ( (*(_BYTE *)(v21 + 96) & 8) != 0 )
          goto LABEL_26;
        if ( (*(_BYTE *)(v22 + 177) & 1) != 0 )
        {
          v24 = *(_QWORD **)(*(_QWORD *)(v21 + 56) + 168LL);
          while ( v24[3] != v21 )
          {
            v24 = (_QWORD *)*v24;
            if ( !v24 )
              goto LABEL_30;
          }
          goto LABEL_26;
        }
LABEL_30:
        v30 = v14;
        sub_7E3660(a1, a2, v21, a3, a4, v14, v31);
        v21 = *(_QWORD *)(v21 + 16);
        v14 = v30;
      }
      while ( v21 );
LABEL_27:
      v13 = *(_QWORD *)(v25 + 16);
      if ( !v13 )
        return v31[0];
    }
  }
  return v10;
}
