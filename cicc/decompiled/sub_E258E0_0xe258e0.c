// Function: sub_E258E0
// Address: 0xe258e0
//
unsigned __int64 __fastcall sub_E258E0(__int64 a1, size_t *a2)
{
  char v3; // al
  __int64 v4; // rdx
  unsigned __int64 v6; // r14
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rbx
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v12; // rsi
  char v13; // dl
  __int64 v14; // rsi
  __int64 v15; // rcx
  const char *v16; // r8
  __int64 *v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax

  v3 = sub_E20730(a2, 3u, "?_7");
  v4 = 1;
  if ( v3 )
    return sub_E29300(a1, a2, v4);
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_8") )
  {
    v4 = 2;
    return sub_E29300(a1, a2, v4);
  }
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_9") )
    return sub_E26E30(a1, a2);
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_A") )
    goto LABEL_19;
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_B") )
  {
    v8 = 0;
    return sub_E26700(a1, a2, v8);
  }
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_C") )
    return sub_E21F00(a1, a2);
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_P") )
  {
LABEL_19:
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  if ( (unsigned __int8)sub_E20730(a2, 4u, "?_R0") )
  {
    v9 = sub_E27700(a1, a2, 2);
    if ( !*(_BYTE *)(a1 + 8) && (unsigned __int8)sub_E20730(a2, 2u, "@8") && !*a2 )
    {
      v10 = *(_QWORD **)(a1 + 16);
      v6 = (*v10 + v10[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v10[1] = v6 - *v10 + 40;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        v21 = (__int64 *)sub_22077B0(32);
        v22 = v21;
        if ( v21 )
        {
          *v21 = 0;
          v21[1] = 0;
          v21[2] = 0;
          v21[3] = 0;
        }
        v23 = sub_2207820(4096);
        v22[2] = 4096;
        *v22 = v23;
        v6 = v23;
        v24 = *(_QWORD *)(a1 + 16);
        v22[1] = 40;
        v22[3] = v24;
        *(_QWORD *)(a1 + 16) = v22;
      }
      if ( !v6 )
      {
        MEMORY[0x20] = v9;
        BUG();
      }
      *(_QWORD *)(v6 + 32) = 0;
      *(_DWORD *)(v6 + 8) = 27;
      *(_QWORD *)(v6 + 16) = 0;
      *(_QWORD *)v6 = &unk_49E11E0;
      *(_BYTE *)(v6 + 24) = 0;
      *(_QWORD *)(v6 + 32) = v9;
      v11 = *(_QWORD **)(a1 + 16);
      v12 = (*v11 + v11[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v11[1] = v12 - *v11 + 40;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        v17 = (__int64 *)sub_22077B0(32);
        v18 = v17;
        if ( v17 )
        {
          *v17 = 0;
          v17[1] = 0;
          v17[2] = 0;
          v17[3] = 0;
        }
        v19 = sub_2207820(4096);
        v18[2] = 4096;
        *v18 = v19;
        v12 = v19;
        v20 = *(_QWORD *)(a1 + 16);
        v18[1] = 40;
        v18[3] = v20;
        *(_QWORD *)(a1 + 16) = v18;
      }
      if ( !v12 )
      {
        MEMORY[0x18] = 0;
        BUG();
      }
      *(_QWORD *)(v12 + 24) = 0;
      *(_QWORD *)(v12 + 32) = 0;
      *(_DWORD *)(v12 + 8) = 5;
      *(_QWORD *)v12 = &unk_49E0F88;
      *(_QWORD *)(v12 + 16) = 0;
      *(_QWORD *)(v12 + 24) = 22;
      *(_QWORD *)(v12 + 32) = "`RTTI Type Descriptor'";
      *(_QWORD *)(v6 + 16) = sub_E20AE0((__int64 **)(a1 + 16), v12);
      return v6;
    }
    goto LABEL_19;
  }
  if ( (unsigned __int8)sub_E20730(a2, 4u, "?_R1") )
    return sub_E26BC0(a1, a1 + 16, a2);
  if ( (unsigned __int8)sub_E20730(a2, 4u, "?_R2") )
  {
    v14 = a1 + 16;
    v15 = 23;
    v16 = "`RTTI Base Class Array'";
    return sub_E26960(a1, v14, a2, v15, v16);
  }
  if ( (unsigned __int8)sub_E20730(a2, 4u, "?_R3") )
  {
    v14 = a1 + 16;
    v15 = 33;
    v16 = "`RTTI Class Hierarchy Descriptor'";
    return sub_E26960(a1, v14, a2, v15, v16);
  }
  if ( (unsigned __int8)sub_E20730(a2, 4u, "?_R4") )
  {
    v4 = 15;
    return sub_E29300(a1, a2, v4);
  }
  if ( (unsigned __int8)sub_E20730(a2, 3u, "?_S") )
  {
    v4 = 16;
    return sub_E29300(a1, a2, v4);
  }
  if ( (unsigned __int8)sub_E20730(a2, 4u, "?__E") )
  {
    v13 = 0;
  }
  else
  {
    if ( !(unsigned __int8)sub_E20730(a2, 4u, "?__F") )
    {
      v6 = 0;
      v7 = sub_E20730(a2, 4u, "?__J");
      v8 = 1;
      if ( v7 )
        return sub_E26700(a1, a2, v8);
      return v6;
    }
    v13 = 1;
  }
  return sub_E256E0(a1, a2, v13);
}
