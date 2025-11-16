// Function: sub_18C8950
// Address: 0x18c8950
//
__int64 __fastcall sub_18C8950(__int64 a1, int a2)
{
  __int64 *v2; // rax
  __int64 v3; // r13
  unsigned __int8 v5; // al
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r15
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  _QWORD *v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  unsigned __int8 v38; // [rsp+2Bh] [rbp-45h]
  _QWORD v39[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = (__int64 *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  if ( (a1 & 4) != 0 )
    v2 = (__int64 *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v3 = *v2;
  if ( *(_BYTE *)(*v2 + 16) )
    return 1;
  if ( sub_15E4F60(*v2) )
    return 1;
  sub_15E4B50(v3);
  v38 = v5;
  if ( v5 )
  {
    return 1;
  }
  else
  {
    v6 = *(_QWORD *)(v3 + 80);
    v34 = v3 + 72;
    if ( v6 != v3 + 72 )
    {
      while ( 1 )
      {
        if ( !v6 )
          BUG();
        v7 = *(_QWORD *)(v6 + 24);
        v8 = v6 + 16;
        if ( v6 + 16 != v7 )
          break;
LABEL_36:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v34 == v6 )
          return v38;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          if ( !v7 )
            BUG();
          v9 = *(_BYTE *)(v7 - 8);
          v10 = v7 - 24;
          if ( v9 <= 0x17u )
            goto LABEL_13;
          if ( v9 == 78 )
          {
            v11 = v10 | 4;
          }
          else
          {
            if ( v9 != 29 )
              goto LABEL_13;
            v11 = v10 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
          if ( a2 == 3 || !v12 )
            goto LABEL_13;
          v35 = (_QWORD *)(v12 + 56);
          if ( (v11 & 4) != 0 )
            break;
          if ( (unsigned __int8)sub_1560260(v35, -1, 36) )
            goto LABEL_13;
          if ( *(char *)(v12 + 23) >= 0 )
            goto LABEL_65;
          v22 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
          v24 = v23 + v22;
          v25 = 0;
          v33 = v24;
          if ( *(char *)(v12 + 23) < 0 )
            v25 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
          if ( !(unsigned int)((v33 - v25) >> 4) )
          {
LABEL_65:
            v26 = *(_QWORD *)(v12 - 72);
            if ( !*(_BYTE *)(v26 + 16) )
            {
              v39[0] = *(_QWORD *)(v26 + 112);
              if ( (unsigned __int8)sub_1560260(v39, -1, 36) )
                goto LABEL_13;
            }
          }
          if ( (unsigned __int8)sub_1560260(v35, -1, 37) )
            goto LABEL_13;
          if ( *(char *)(v12 + 23) < 0 )
          {
            v27 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
            v29 = v27 + v28;
            if ( *(char *)(v12 + 23) >= 0 )
            {
              v30 = 0;
            }
            else
            {
              v37 = v27 + v28;
              v30 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
              v29 = v37;
            }
            if ( v30 != v29 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v30 + 8LL) <= 1u )
              {
                v30 += 16;
                if ( v29 == v30 )
                  goto LABEL_53;
              }
              goto LABEL_34;
            }
          }
LABEL_53:
          v31 = *(_QWORD *)(v12 - 72);
          if ( *(_BYTE *)(v31 + 16) )
            goto LABEL_34;
LABEL_54:
          v39[0] = *(_QWORD *)(v31 + 112);
          if ( !(unsigned __int8)sub_1560260(v39, -1, 37) )
            goto LABEL_34;
LABEL_13:
          v7 = *(_QWORD *)(v7 + 8);
          if ( v8 == v7 )
            goto LABEL_36;
        }
        if ( (unsigned __int8)sub_1560260((_QWORD *)(v12 + 56), -1, 36) )
          goto LABEL_13;
        if ( *(char *)(v12 + 23) >= 0 )
          goto LABEL_66;
        v13 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
        v15 = v14 + v13;
        v16 = 0;
        v32 = v15;
        if ( *(char *)(v12 + 23) < 0 )
          v16 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
        if ( !(unsigned int)((v32 - v16) >> 4) )
        {
LABEL_66:
          v17 = *(_QWORD *)(v12 - 24);
          if ( !*(_BYTE *)(v17 + 16) )
          {
            v39[0] = *(_QWORD *)(v17 + 112);
            if ( (unsigned __int8)sub_1560260(v39, -1, 36) )
              goto LABEL_13;
          }
        }
        if ( (unsigned __int8)sub_1560260(v35, -1, 37) )
          goto LABEL_13;
        if ( *(char *)(v12 + 23) < 0 )
        {
          v18 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
          v20 = v18 + v19;
          if ( *(char *)(v12 + 23) >= 0 )
          {
            v21 = 0;
          }
          else
          {
            v36 = v18 + v19;
            v21 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
            v20 = v36;
          }
          if ( v21 != v20 )
          {
            while ( *(_DWORD *)(*(_QWORD *)v21 + 8LL) <= 1u )
            {
              v21 += 16;
              if ( v20 == v21 )
                goto LABEL_57;
            }
            goto LABEL_34;
          }
        }
LABEL_57:
        v31 = *(_QWORD *)(v12 - 24);
        if ( !*(_BYTE *)(v31 + 16) )
          goto LABEL_54;
LABEL_34:
        if ( (unsigned __int8)sub_18C8950(v11, (unsigned int)(a2 + 1)) )
          return 1;
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          goto LABEL_36;
      }
    }
  }
  return v38;
}
