// Function: sub_33AB4F0
// Address: 0x33ab4f0
//
__int64 __fastcall sub_33AB4F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  _BYTE *v13; // r13
  __int64 v14; // rbx
  char *v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int8 v20; // r14
  bool v21; // bl
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int8 v25; // r14
  bool v26; // bl
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // [rsp-10h] [rbp-70h]
  __int64 v31[10]; // [rsp+10h] [rbp-50h] BYREF

  if ( *(char *)(a2 + 7) < 0 )
  {
    v4 = sub_BD2BC0(a2);
    v6 = v4 + v5;
    if ( *(char *)(a2 + 7) < 0 )
      v6 -= sub_BD2BC0(a2);
    v7 = v6 >> 4;
    if ( (_DWORD)v7 )
    {
      v8 = 0;
      v9 = 16LL * (unsigned int)v7;
      while ( 1 )
      {
        v10 = 0;
        if ( *(char *)(a2 + 7) < 0 )
          v10 = sub_BD2BC0(a2);
        v11 = v8 + v10;
        v12 = *(_QWORD **)v11;
        if ( **(_QWORD **)v11 == 7
          && *((_DWORD *)v12 + 4) == 1634890864
          && *((_WORD *)v12 + 10) == 29813
          && *((_BYTE *)v12 + 22) == 104 )
        {
          break;
        }
        v8 += 16;
        if ( v8 == v9 )
          goto LABEL_14;
      }
      v3 = (__int64 *)(a2 + 32 * (*(unsigned int *)(v11 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    }
  }
LABEL_14:
  v13 = *(_BYTE **)(a2 - 32);
  v14 = *v3;
  v15 = (char *)v3[4];
  if ( *v13 == 8
    && (v16 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL)),
        (unsigned __int8)sub_AC4540((__int64)v13, v14, v15, v16)) )
  {
    v25 = sub_B49200(a2);
    v26 = sub_B49220(a2);
    v27 = sub_338B750(a1, *((_QWORD *)v13 - 16));
    sub_33A7C00(a1, (unsigned __int8 *)a2, v27, v28, v26, v25, a3, 0);
  }
  else
  {
    v17 = *(_QWORD **)(v14 + 24);
    if ( *(_DWORD *)(v14 + 32) > 0x40u )
      v17 = (_QWORD *)*v17;
    v31[0] = (__int64)v17;
    v18 = sub_338B750(a1, (__int64)v15);
    v31[2] = v19;
    v31[1] = v18;
    v20 = sub_B49200(a2);
    v21 = sub_B49220(a2);
    v22 = sub_338B750(a1, (__int64)v13);
    sub_33A7C00(a1, (unsigned __int8 *)a2, v22, v23, v21, v20, a3, v31);
  }
  return v29;
}
