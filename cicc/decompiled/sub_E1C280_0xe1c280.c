// Function: sub_E1C280
// Address: 0xe1c280
//
__int64 __fastcall sub_E1C280(__int64 a1, _BYTE *a2)
{
  char *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rdx
  const char *v6; // r13
  size_t v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  char v13; // dl
  _BYTE *v14; // rax
  _BYTE *v15; // rdx
  char v16; // r13
  char v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  char v23; // dl
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r12
  char v29; // dl
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r12

  v2 = sub_E0E160(a1);
  if ( v2 )
  {
    v5 = (unsigned __int8)v2[2];
    if ( (_BYTE)v5 != 8 )
    {
      if ( (unsigned __int8)v5 <= 0xAu && ((_BYTE)v5 != 4 || (v2[3] & 1) != 0) )
      {
        v6 = (const char *)*((_QWORD *)v2 + 1);
        v7 = strlen(v6);
        result = sub_E0E790(a1 + 816, 32, v8, v9, v10, v11);
        if ( result )
        {
          v13 = *(_BYTE *)(result + 10);
          *(_QWORD *)(result + 16) = v7;
          *(_WORD *)(result + 8) = 16392;
          *(_QWORD *)(result + 24) = v6;
          *(_BYTE *)(result + 10) = v13 & 0xF0 | 5;
          *(_QWORD *)result = &unk_49DEFA8;
        }
        return result;
      }
      return 0;
    }
    v16 = *(_BYTE *)(a1 + 777);
    v17 = *(_BYTE *)(a1 + 776);
    *(_BYTE *)(a1 + 776) = 0;
    if ( v16 )
    {
      *(_BYTE *)(a1 + 777) = 1;
      v22 = sub_E1AEA0(a1, (__int64)a2, v5, v3, v4);
      if ( !v22 )
        goto LABEL_27;
      if ( a2 )
        goto LABEL_16;
    }
    else
    {
      if ( a2 )
      {
        *(_BYTE *)(a1 + 777) = 1;
        v22 = sub_E1AEA0(a1, (__int64)a2, v5, v3, v4);
        if ( v22 )
        {
LABEL_16:
          *a2 = 1;
          goto LABEL_17;
        }
LABEL_27:
        result = 0;
LABEL_19:
        *(_BYTE *)(a1 + 777) = v16;
        *(_BYTE *)(a1 + 776) = v17;
        return result;
      }
      v22 = sub_E1AEA0(a1, 0, v5, v3, v4);
      if ( !v22 )
        goto LABEL_27;
    }
LABEL_17:
    result = sub_E0E790(a1 + 816, 24, v18, v19, v20, v21);
    if ( result )
    {
      v23 = *(_BYTE *)(result + 10);
      *(_QWORD *)(result + 16) = v22;
      *(_WORD *)(result + 8) = 16388;
      *(_BYTE *)(result + 10) = v23 & 0xF0 | 5;
      *(_QWORD *)result = &unk_49DEEE8;
    }
    goto LABEL_19;
  }
  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "li") )
  {
    v28 = sub_E12DE0((__int64 *)a1);
    if ( !v28 )
      return 0;
    result = sub_E0E790(a1 + 816, 24, v24, v25, v26, v27);
    if ( result )
    {
      v29 = *(_BYTE *)(result + 10);
      *(_QWORD *)(result + 16) = v28;
      *(_WORD *)(result + 8) = 16404;
      *(_BYTE *)(result + 10) = v29 & 0xF0 | 5;
      *(_QWORD *)result = &unk_49DF5A8;
    }
  }
  else
  {
    v14 = *(_BYTE **)a1;
    v15 = *(_BYTE **)(a1 + 8);
    if ( *(_BYTE **)a1 == v15 )
      return 0;
    if ( *v14 != 118 )
      return 0;
    *(_QWORD *)a1 = v14 + 1;
    if ( v15 == v14 + 1 )
      return 0;
    if ( (unsigned __int8)(v14[1] - 48) > 9u )
      return 0;
    *(_QWORD *)a1 = v14 + 2;
    v34 = sub_E12DE0((__int64 *)a1);
    if ( !v34 )
      return 0;
    result = sub_E0E790(a1 + 816, 24, v30, v31, v32, v33);
    if ( result )
    {
      *(_QWORD *)(result + 16) = v34;
      *(_WORD *)(result + 8) = 16388;
      *(_BYTE *)(result + 10) = *(_BYTE *)(result + 10) & 0xF0 | 5;
      *(_QWORD *)result = &unk_49DEEE8;
    }
  }
  return result;
}
