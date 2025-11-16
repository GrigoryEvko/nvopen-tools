// Function: sub_E20280
// Address: 0xe20280
//
__int64 __fastcall sub_E20280(__int64 a1)
{
  int v1; // r12d
  __int64 v2; // rbx
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned __int8 *v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  unsigned __int8 *v13; // rax
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int8 *v26; // rax
  void *v27; // r14
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  char v33; // al
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r13
  unsigned __int8 *v38; // rax
  unsigned __int8 *v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // r14
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  char v50; // dl
  char v51; // [rsp+7h] [rbp-49h]
  __int64 v52; // [rsp+8h] [rbp-48h]
  void *v53; // [rsp+8h] [rbp-48h]
  __int64 v54[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = sub_E0E0E0(a1);
  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Do") )
  {
    v2 = sub_E0FD70(a1 + 816, "noexcept");
    if ( !v2 )
      return 0;
    goto LABEL_3;
  }
  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "DO") )
  {
    v12 = sub_E18BB0(a1);
    if ( !v12 )
      return 0;
    v13 = *(unsigned __int8 **)a1;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
      return 0;
    if ( *v13 != 69 )
      return 0;
    *(_QWORD *)a1 = v13 + 1;
    v14 = sub_E0E790(a1 + 816, 24, v8, v9, v10, v11);
    v2 = v14;
    if ( !v14 )
      return 0;
    *(_WORD *)(v14 + 8) = 16401;
    v15 = *(_BYTE *)(v14 + 10);
    *(_QWORD *)(v2 + 16) = v12;
    *(_BYTE *)(v2 + 10) = v15 & 0xF0 | 5;
    *(_QWORD *)v2 = &unk_49DF428;
    goto LABEL_3;
  }
  v16 = 2;
  if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, &unk_3F7C044) )
  {
    v2 = 0;
LABEL_3:
    sub_E0F5E0((const void **)a1, 2u, &unk_3F7C047);
    v5 = *(unsigned __int8 **)a1;
    v6 = *(_QWORD *)(a1 + 8);
    if ( *(_QWORD *)a1 != v6 && *v5 == 70 )
    {
      *(_QWORD *)a1 = v5 + 1;
      if ( (unsigned __int8 *)v6 != v5 + 1 && v5[1] == 89 )
        *(_QWORD *)a1 = v5 + 2;
      v37 = sub_E1AEA0(a1, (__int64)(v5 + 1), v6, v3, v4);
      if ( v37 )
      {
        v38 = *(unsigned __int8 **)a1;
        v39 = *(unsigned __int8 **)(a1 + 8);
        v52 = (__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3;
        while ( 1 )
        {
          if ( v39 == v38 )
            goto LABEL_30;
LABEL_28:
          v40 = *v38;
          if ( (_BYTE)v40 == 69 )
          {
            v51 = 0;
            *(_QWORD *)a1 = v38 + 1;
            goto LABEL_36;
          }
          if ( (_BYTE)v40 != 118 )
            break;
          *(_QWORD *)a1 = ++v38;
        }
        while ( 1 )
        {
LABEL_30:
          if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "RE") )
          {
            v51 = 1;
            goto LABEL_36;
          }
          if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "OE") )
            break;
          result = sub_E1AEA0(a1, 2, v40, v34, v35);
          v54[0] = result;
          if ( !result )
            return result;
          sub_E18380(a1 + 16, v54, v41, v42, v43, v44);
          v38 = *(unsigned __int8 **)a1;
          v39 = *(unsigned __int8 **)(a1 + 8);
          if ( v39 != *(unsigned __int8 **)a1 )
            goto LABEL_28;
        }
        v51 = 2;
LABEL_36:
        v53 = sub_E11E80((_QWORD *)a1, v52, v40, v34, v35, v36);
        v46 = v45;
        result = sub_E0E790(a1 + 816, 56, v45, v47, v48, v49);
        if ( result )
        {
          *(_QWORD *)(result + 16) = v37;
          *(_WORD *)(result + 8) = 16;
          v50 = *(_BYTE *)(result + 10);
          *(_QWORD *)(result + 24) = v53;
          *(_QWORD *)(result + 32) = v46;
          *(_DWORD *)(result + 40) = v1;
          *(_BYTE *)(result + 10) = v50 & 0xF0 | 1;
          *(_BYTE *)(result + 44) = v51;
          *(_QWORD *)(result + 48) = v2;
          *(_QWORD *)result = &unk_49DF3C8;
        }
        return result;
      }
    }
    return 0;
  }
  v21 = (__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3;
  while ( 1 )
  {
    v26 = *(unsigned __int8 **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v26 == 69 )
      break;
    result = sub_E1AEA0(a1, v16, v17, v18, v19);
    v54[0] = result;
    if ( !result )
      return result;
    v16 = (__int64)v54;
    sub_E18380(a1 + 16, v54, v22, v23, v24, v25);
  }
  *(_QWORD *)a1 = v26 + 1;
  v27 = sub_E11E80((_QWORD *)a1, v21, v17, v18, v19, v20);
  v29 = v28;
  v2 = sub_E0E790(a1 + 816, 32, v28, v30, v31, v32);
  result = 0;
  if ( v2 )
  {
    v33 = *(_BYTE *)(v2 + 10);
    *(_QWORD *)(v2 + 16) = v27;
    *(_WORD *)(v2 + 8) = 16402;
    *(_QWORD *)(v2 + 24) = v29;
    *(_BYTE *)(v2 + 10) = v33 & 0xF0 | 5;
    *(_QWORD *)v2 = &unk_49DF488;
    goto LABEL_3;
  }
  return result;
}
