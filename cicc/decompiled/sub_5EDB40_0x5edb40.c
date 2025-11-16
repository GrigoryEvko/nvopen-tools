// Function: sub_5EDB40
// Address: 0x5edb40
//
__int64 __fastcall sub_5EDB40(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  char v5; // al
  _BOOL4 v6; // r14d
  __int64 v7; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r9
  char v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rdx
  char v19; // al
  __int64 v20; // rdi
  char v21; // al
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rax
  _QWORD *v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rdi
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  unsigned int v33[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = *(_BYTE *)(a3 + 133);
  *a4 = 0;
  v6 = (v5 & 4) != 0;
  if ( (*(_BYTE *)(a1 + 17) & 0x20) != 0 )
    return sub_647630(10, a1, unk_4F04C5C, v6);
  v11 = sub_5E69C0(a1, a2);
  if ( !v11 )
    return sub_647630(10, a1, unk_4F04C5C, v6);
  v29 = v11;
  v12 = sub_5EDAE0(v11, a3, 0, 0);
  v13 = v29;
  v7 = v12;
  if ( v12 )
  {
    v14 = *(_BYTE *)(v12 + 80);
    v15 = v7;
    if ( v14 == 16 )
    {
      v15 = **(_QWORD **)(v7 + 88);
      v14 = *(_BYTE *)(v15 + 80);
    }
    if ( v14 == 24 )
      v15 = *(_QWORD *)(v15 + 88);
    if ( (*(_BYTE *)(v15 + 104) & 1) != 0 )
    {
      v17 = sub_8796F0(v15);
      v13 = v29;
    }
    else
    {
      v16 = *(_QWORD *)(v15 + 88);
      if ( *(_BYTE *)(v15 + 80) == 20 )
        v16 = *(_QWORD *)(v16 + 176);
      v17 = (*(_BYTE *)(v16 + 208) & 4) != 0;
    }
    if ( v17 )
    {
      v6 = 1;
    }
    else
    {
      if ( *(_BYTE *)(v7 + 80) != 16 || (*(_BYTE *)(v7 + 96) & 4) == 0 )
        return v7;
      if ( (*(_BYTE *)(a1 + 16) & 0x10) != 0 )
      {
        v18 = **(_QWORD **)(v7 + 88);
        v19 = *(_BYTE *)(v18 + 80);
        if ( v19 == 24 )
          v19 = *(_BYTE *)(*(_QWORD *)(v18 + 88) + 80LL);
        if ( v19 == 10 )
        {
          v25 = (_QWORD *)(*(_QWORD *)(**(_QWORD **)(v7 + 64) + 96LL) + 40LL);
          v26 = (_QWORD *)*v25;
          if ( *v25 )
          {
            while ( 1 )
            {
              v27 = v26[1];
              v28 = v26;
              v26 = (_QWORD *)*v26;
              if ( v7 == v27 )
                break;
              v25 = v28;
              if ( !v26 )
                goto LABEL_21;
            }
            *v25 = v26;
            *v28 = 0;
            v32 = v13;
            sub_878490();
            v13 = v32;
          }
        }
      }
LABEL_21:
      if ( v13 == v7 || (v20 = v7, v30 = v13, v7 = 0, sub_879190(v20, v13), v13 = v30, !*(_QWORD *)(v30 + 88)) )
      {
        sub_881DB0(v13);
        return sub_647630(10, a1, unk_4F04C5C, v6);
      }
    }
  }
  v21 = *(_BYTE *)(v13 + 80);
  v22 = v13;
  if ( v21 == 16 )
  {
    v22 = **(_QWORD **)(v13 + 88);
    v21 = *(_BYTE *)(v22 + 80);
  }
  if ( v21 == 24 )
  {
    v22 = *(_QWORD *)(v22 + 88);
    v21 = *(_BYTE *)(v22 + 80);
  }
  if ( v21 == 2 && (v24 = *(_QWORD *)(v22 + 88)) != 0 && *(_BYTE *)(v24 + 173) == 12 )
  {
    v6 = 1;
  }
  else if ( v6 || (v31 = v13, v23 = sub_8E0850(v13, *(_QWORD *)(a3 + 288), a3, v33), v13 = v31, v23) )
  {
    v7 = sub_887500(10, a1, (*(_BYTE *)(a3 + 560) & 2) != 0, v13, a4);
  }
  else
  {
    v6 = 1;
    sub_6851C0(v33[0], a1 + 8);
    *(_BYTE *)(a1 + 17) |= 0x20u;
    *(_QWORD *)(a1 + 24) = 0;
  }
  if ( !v7 )
    return sub_647630(10, a1, unk_4F04C5C, v6);
  return v7;
}
