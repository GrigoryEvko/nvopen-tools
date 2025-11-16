// Function: sub_A5F430
// Address: 0xa5f430
//
__int64 __fastcall sub_A5F430(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r8
  int v10; // eax
  unsigned __int8 v11; // al
  __int64 v12; // rbx
  __int64 *v13; // rbx
  char v14; // r14
  _WORD *v15; // rdx
  __int64 v16; // r15
  _DWORD *v17; // rdx
  __int64 *v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  char v21; // [rsp+18h] [rbp-48h]
  const char *v22; // [rsp+20h] [rbp-40h]
  __int64 *v23; // [rsp+28h] [rbp-38h]

  v4 = a1;
  v5 = a2 - 16;
  sub_904010(a1, "!GenericDINode(");
  v20 = a1;
  v21 = 1;
  v22 = ", ";
  v23 = a3;
  sub_A53560(&v20, a2);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = **(_QWORD **)(a2 - 32);
    if ( v7 )
    {
LABEL_3:
      v7 = sub_B91420(v7, a2);
      v9 = v8;
      goto LABEL_4;
    }
  }
  else
  {
    v7 = *(_QWORD *)(v5 - 8LL * ((v6 >> 2) & 0xF));
    if ( v7 )
      goto LABEL_3;
  }
  v9 = 0;
LABEL_4:
  sub_A53660(&v20, "header", 6u, v7, v9, 1);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v10 = *(_DWORD *)(a2 - 24);
  else
    v10 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  if ( v10 != 1 )
  {
    if ( v21 )
      v21 = 0;
    else
      a1 = sub_904010(a1, v22);
    sub_904010(a1, "operands: {");
    v11 = *(_BYTE *)(a2 - 16);
    if ( (v11 & 2) != 0 )
    {
      v12 = *(_QWORD *)(a2 - 32);
      v19 = (__int64 *)(v12 + 8LL * *(unsigned int *)(a2 - 24));
    }
    else
    {
      v12 = v5 - 8LL * ((v11 >> 2) & 0xF);
      v19 = (__int64 *)(v12 + 8LL * ((*(_WORD *)(a2 - 16) >> 6) & 0xF));
    }
    v13 = (__int64 *)(v12 + 8);
    if ( v13 != v19 )
    {
      v14 = 1;
      while ( 1 )
      {
        if ( v14 )
        {
          v16 = *v13;
          v14 = 0;
          if ( !*v13 )
            goto LABEL_20;
LABEL_16:
          sub_A5C090(v4, v16, a3);
          (*(void (__fastcall **)(__int64 *, __int64))*a3)(a3, v16);
LABEL_17:
          if ( v19 == ++v13 )
            break;
        }
        else
        {
          v15 = *(_WORD **)(v4 + 32);
          if ( *(_QWORD *)(v4 + 24) - (_QWORD)v15 <= 1u )
          {
            sub_CB6200(v4, ", ", 2);
          }
          else
          {
            *v15 = 8236;
            *(_QWORD *)(v4 + 32) += 2LL;
          }
          v16 = *v13;
          if ( *v13 )
            goto LABEL_16;
LABEL_20:
          v17 = *(_DWORD **)(v4 + 32);
          if ( *(_QWORD *)(v4 + 24) - (_QWORD)v17 <= 3u )
          {
            sub_CB6200(v4, "null", 4);
            goto LABEL_17;
          }
          *v17 = 1819047278;
          ++v13;
          *(_QWORD *)(v4 + 32) += 4LL;
          if ( v19 == v13 )
            break;
        }
      }
    }
    sub_904010(v4, "}");
  }
  return sub_904010(v4, ")");
}
