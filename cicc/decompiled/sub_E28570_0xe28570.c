// Function: sub_E28570
// Address: 0xe28570
//
unsigned __int64 __fastcall sub_E28570(__int64 a1, __int64 *a2, char a3)
{
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  _BYTE *v8; // rdx
  char v10; // al
  int v11; // edx
  char v12; // bl
  __int64 v13; // rax
  _BYTE *v14; // rcx
  __int64 *v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax

  v4 = *(_QWORD **)(a1 + 16);
  v5 = (*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v5 - *v4 + 64;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v15 = (__int64 *)sub_22077B0(32);
    v16 = v15;
    if ( v15 )
    {
      *v15 = 0;
      v15[1] = 0;
      v15[2] = 0;
      v15[3] = 0;
    }
    v17 = sub_2207820(4096);
    v16[2] = 4096;
    *v16 = v17;
    v6 = v17;
    v18 = *(_QWORD *)(a1 + 16);
    v16[1] = 64;
    v16[3] = v18;
    *(_QWORD *)(a1 + 16) = v16;
    if ( !v6 )
    {
LABEL_3:
      if ( !a3 )
        goto LABEL_4;
      goto LABEL_10;
    }
  }
  else
  {
    v6 = 0;
    if ( !v5 )
      goto LABEL_3;
    v6 = v5;
  }
  *(_BYTE *)(v6 + 12) = 0;
  *(_DWORD *)(v6 + 8) = 3;
  *(_BYTE *)(v6 + 20) = 0;
  *(_QWORD *)v6 = &unk_49E1078;
  *(_DWORD *)(v6 + 16) = 0;
  *(_WORD *)(v6 + 22) = 8;
  *(_DWORD *)(v6 + 24) = 0;
  *(_QWORD *)(v6 + 32) = 0;
  *(_BYTE *)(v6 + 40) = 0;
  *(_QWORD *)(v6 + 48) = 0;
  *(_BYTE *)(v6 + 56) = 0;
  if ( !a3 )
    goto LABEL_4;
LABEL_10:
  v10 = sub_E24310(a1, a2);
  v11 = 0;
  *(_BYTE *)(v6 + 12) = v10;
  v12 = v10;
  v13 = *a2;
  if ( *a2 )
  {
    v14 = (_BYTE *)a2[1];
    if ( *v14 == 71 )
    {
      v11 = 1;
      a2[1] = (__int64)(v14 + 1);
      *a2 = v13 - 1;
      v12 = *(_BYTE *)(v6 + 12);
    }
    else if ( *v14 == 72 )
    {
      v11 = 2;
      a2[1] = (__int64)(v14 + 1);
      *a2 = v13 - 1;
      v12 = *(_BYTE *)(v6 + 12);
    }
  }
  *(_DWORD *)(v6 + 24) = v11;
  *(_BYTE *)(v6 + 12) = sub_E22E40(a1, a2) | v12;
LABEL_4:
  *(_BYTE *)(v6 + 20) = sub_E22DC0(a1, a2);
  v7 = *a2;
  if ( *a2 && (v8 = (_BYTE *)a2[1], *v8 == 64) )
  {
    a2[1] = (__int64)(v8 + 1);
    *a2 = v7 - 1;
  }
  else
  {
    *(_QWORD *)(v6 + 32) = sub_E27700(a1, a2, 2);
  }
  *(_QWORD *)(v6 + 48) = sub_E28110(a1, a2, (_BYTE *)(v6 + 40));
  *(_BYTE *)(v6 + 56) = sub_E22F00(a1, (size_t *)a2);
  return v6;
}
