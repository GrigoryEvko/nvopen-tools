// Function: sub_E256E0
// Address: 0xe256e0
//
__int64 __fastcall sub_E256E0(__int64 a1, size_t *a2, char a3)
{
  _QWORD *v4; // rax
  unsigned __int64 v5; // r12
  char v6; // bl
  size_t v7; // rax
  _BYTE *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  size_t v12; // rdx
  _BYTE *v13; // rcx
  int v14; // ebx
  size_t v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax

  v4 = *(_QWORD **)(a1 + 16);
  v5 = (*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v5 - *v4 + 48;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v16 = (__int64 *)sub_22077B0(32);
    v17 = v16;
    if ( v16 )
    {
      *v16 = 0;
      v16[1] = 0;
      v16[2] = 0;
      v16[3] = 0;
    }
    v18 = sub_2207820(4096);
    v17[2] = 4096;
    *v17 = v18;
    v5 = v18;
    v19 = *(_QWORD *)(a1 + 16);
    v17[1] = 48;
    v17[3] = v19;
    *(_QWORD *)(a1 + 16) = v17;
  }
  if ( !v5 )
  {
    MEMORY[0x28] = 0;
    BUG();
  }
  *(_BYTE *)(v5 + 40) = 0;
  v6 = 0;
  *(_DWORD *)(v5 + 8) = 10;
  *(_BYTE *)(v5 + 40) = a3;
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)v5 = &unk_49E0F60;
  *(_QWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 32) = 0;
  v7 = *a2;
  v8 = (_BYTE *)a2[1];
  if ( *a2 && *v8 == 63 )
  {
    v6 = 1;
    a2[1] = (size_t)(v8 + 1);
    *a2 = v7 - 1;
  }
  v9 = sub_E25660(a1, a2);
  v10 = v9;
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  if ( *(_DWORD *)(v9 + 8) == 27 )
  {
    *(_QWORD *)(v5 + 24) = v9;
    v12 = *a2;
    v13 = (_BYTE *)a2[1];
    v14 = (v6 & 1) + 1;
    v15 = *a2;
    while ( v15 && *v13 == 64 )
    {
      --v15;
      a2[1] = (size_t)++v13;
      *a2 = v15;
      if ( v14 <= (int)v12 - (int)v15 )
      {
        v10 = sub_E28950(a1, a2);
        if ( v10 )
          goto LABEL_8;
        return v10;
      }
    }
    goto LABEL_14;
  }
  if ( v6 )
  {
LABEL_14:
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  *(_QWORD *)(v5 + 32) = *(_QWORD *)(v9 + 16);
LABEL_8:
  *(_QWORD *)(v10 + 16) = sub_E20AE0((__int64 **)(a1 + 16), v5);
  return v10;
}
