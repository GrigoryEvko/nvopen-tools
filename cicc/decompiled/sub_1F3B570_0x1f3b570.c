// Function: sub_1F3B570
// Address: 0x1f3b570
//
__int64 __fastcall sub_1F3B570(__int64 *a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 *v9; // rax
  __int64 v10; // rdx
  __int16 v11; // ax
  __int64 v12; // r14
  __int64 (*v13)(); // rax
  __int64 (*v14)(); // rax
  __int64 v15; // r12
  char v17; // al
  char v18; // al
  __int64 v19; // rdx
  __int64 *v20; // rbx
  __int64 *v21; // r13
  __int64 v22; // rdx
  int v23; // eax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  int v29[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v9 = *(__int16 **)(a2 + 16);
  v29[0] = 0;
  v10 = *a1;
  v11 = *v9;
  v12 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 56LL);
  if ( (v11 & 0xFFFB) == 0x13 || v11 == 21 )
  {
    v13 = *(__int64 (**)())(v10 + 48);
    if ( v13 != sub_1E1C810 )
    {
      v27 = a6;
      v28 = *(_QWORD *)(a2 + 24);
      v23 = ((__int64 (__fastcall *)(__int64 *, __int64, int *))v13)(a1, a5, v29);
      a6 = v27;
      if ( v23 )
      {
        v24 = sub_1F3A290(v12, a2, a3, a4, v29[0], a1);
        v15 = (__int64)v24;
        if ( v24 )
        {
          sub_1DD5BA0((__int64 *)(v28 + 16), (__int64)v24);
          v25 = *(_QWORD *)a2;
          v26 = *(_QWORD *)v15;
          *(_QWORD *)(v15 + 8) = a2;
          v25 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v15 = v25 | v26 & 7;
          *(_QWORD *)(v25 + 8) = v15;
          *(_QWORD *)a2 = v15 | *(_QWORD *)a2 & 7LL;
          goto LABEL_8;
        }
        return 0;
      }
      v10 = *a1;
    }
  }
  v14 = *(__int64 (**)())(v10 + 512);
  if ( v14 == sub_1F39490 )
    return 0;
  v15 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, char *, __int64, __int64, __int64, __int64))v14)(
          a1,
          v12,
          a2,
          a3,
          a4,
          a2,
          a5,
          a6);
  if ( !v15 )
    return 0;
LABEL_8:
  v17 = *(_BYTE *)(a2 + 49);
  if ( v17 )
  {
    v19 = *(_QWORD *)(a2 + 56);
    *(_BYTE *)(v15 + 49) = v17;
    *(_QWORD *)(v15 + 56) = v19;
    v20 = *(__int64 **)(a5 + 56);
    v21 = &v20[*(unsigned __int8 *)(a5 + 49)];
    while ( v20 != v21 )
    {
      v22 = *v20++;
      sub_1E15C90(v15, v12, v22);
    }
  }
  else
  {
    v18 = *(_BYTE *)(a5 + 49);
    *(_QWORD *)(v15 + 56) = *(_QWORD *)(a5 + 56);
    *(_BYTE *)(v15 + 49) = v18;
  }
  return v15;
}
