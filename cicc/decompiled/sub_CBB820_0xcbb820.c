// Function: sub_CBB820
// Address: 0xcbb820
//
char *__fastcall sub_CBB820(__int64 a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // r13
  char *v8; // rdi
  __int64 v9; // rdx
  int v10; // ebx
  unsigned __int64 v11; // rcx
  int v12; // r15d
  int v13; // r10d
  __int64 v14; // rax
  __int64 v15; // r14
  int v16; // ebx
  __int64 v17; // r12
  __int64 v18; // rax
  int v19; // eax
  int v20; // eax
  int v21; // r8d
  char *v22; // rax
  char *v23; // r15
  int v24; // eax
  int v25; // eax
  int v26; // edi
  int v27; // eax
  int v29[2]; // [rsp+0h] [rbp-80h]
  int v31; // [rsp+10h] [rbp-70h]
  unsigned __int64 v32; // [rsp+10h] [rbp-70h]
  int v33; // [rsp+10h] [rbp-70h]
  unsigned __int64 v34; // [rsp+10h] [rbp-70h]
  unsigned __int64 v35; // [rsp+10h] [rbp-70h]
  unsigned __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  unsigned __int64 v38; // [rsp+20h] [rbp-60h]
  char *v39; // [rsp+28h] [rbp-58h]
  char *v41; // [rsp+38h] [rbp-48h]
  int v42; // [rsp+44h] [rbp-3Ch]
  char *v43; // [rsp+48h] [rbp-38h]

  v5 = a5;
  v6 = a4;
  v7 = *(_QWORD *)a1;
  v43 = a2;
  if ( a4 >= a5 )
  {
LABEL_6:
    v10 = 128;
    v38 = *(_QWORD *)(a1 + 96);
    if ( *(char **)(a1 + 32) != v43 )
      v10 = *(v43 - 1);
    v41 = 0;
    v37 = 1LL << a5;
    v11 = sub_CBAFE0(v7, v6, a5, 1LL << v6, 132, 1LL << v6);
    v39 = *(char **)(a1 + 40);
    if ( v39 == v43 )
      goto LABEL_36;
LABEL_9:
    v42 = *v43;
    if ( v10 == 10 )
    {
      v12 = *(_DWORD *)(v7 + 40) & 8;
      if ( !v12 )
      {
        while ( 1 )
        {
          v34 = v11;
          v25 = isalnum((unsigned __int8)v10);
          v11 = v34;
          if ( v25 )
            goto LABEL_46;
          if ( v10 != 95 )
          {
            v26 = (unsigned __int8)v10;
            if ( v42 != 128 )
            {
LABEL_21:
              v32 = v11;
              v19 = isalnum((unsigned __int8)v42);
              v11 = v32;
              if ( v42 == 95 || v19 )
              {
                if ( v10 == 128
                  || (*(_QWORD *)v29 = v32,
                      v33 = v19,
                      v20 = isalnum((unsigned __int8)v10),
                      v11 = *(_QWORD *)v29,
                      v10 != 95)
                  && !v20
                  || v42 == 95
                  || (v21 = 134, v33) )
                {
                  v21 = 133;
                }
LABEL_29:
                v11 = sub_CBAFE0(v7, v6, v5, v11, v21, v11);
LABEL_30:
                v10 = v42;
                goto LABEL_31;
              }
              goto LABEL_60;
            }
            goto LABEL_45;
          }
          while ( 2 )
          {
            v21 = 134;
            if ( v12 == 130 )
              goto LABEL_29;
            v10 = 128;
            if ( v42 != 128 )
            {
              v36 = v11;
              v27 = isalnum((unsigned __int8)v42);
              v11 = v36;
              if ( v42 != 95 )
              {
                v21 = 134;
                if ( !v27 )
                  goto LABEL_29;
              }
              goto LABEL_30;
            }
LABEL_31:
            v22 = v41;
            v23 = v43;
            if ( (v11 & v37) != 0 )
              v22 = v43;
            v41 = v22;
            if ( v11 == v38 || v43 == a3 )
              return v41;
            ++v43;
            v11 = sub_CBAFE0(v7, v6, v5, v11, v10, v38);
            if ( v39 != v23 + 1 )
              goto LABEL_9;
LABEL_36:
            if ( v10 == 10 )
            {
              v12 = *(_DWORD *)(v7 + 40) & 8;
              if ( !v12 )
              {
                if ( (*(_BYTE *)(a1 + 8) & 2) == 0 )
                  goto LABEL_78;
LABEL_77:
                v42 = 128;
                goto LABEL_18;
              }
              v13 = *(_DWORD *)(v7 + 76);
              if ( (*(_DWORD *)(a1 + 8) & 2) != 0 )
              {
LABEL_83:
                v12 = 129;
                if ( v13 > 0 )
                {
                  v42 = 128;
                  goto LABEL_15;
                }
                goto LABEL_77;
              }
            }
            else
            {
              v24 = *(_DWORD *)(a1 + 8);
              if ( v10 != 128 || (v24 & 1) != 0 )
              {
                if ( (v24 & 2) != 0 )
                {
                  v42 = 128;
                  goto LABEL_40;
                }
LABEL_78:
                v42 = 128;
                v13 = *(_DWORD *)(v7 + 80);
LABEL_14:
                v12 = 130;
                if ( v13 <= 0 )
                  goto LABEL_41;
LABEL_15:
                v14 = v5;
                v31 = v10;
                v15 = v6;
                v16 = v13;
                v17 = v14;
                do
                {
                  v11 = sub_CBAFE0(v7, v15, v17, v11, v12, v11);
                  --v16;
                }
                while ( v16 );
                v18 = v17;
                v10 = v31;
                v6 = v15;
                v5 = v18;
LABEL_18:
                if ( v12 != 129 )
                  goto LABEL_41;
LABEL_19:
                if ( v42 != 128 )
                {
LABEL_20:
                  v12 = 129;
                  goto LABEL_21;
                }
                v12 = 129;
LABEL_60:
                if ( v10 == 128 )
                  goto LABEL_30;
                v26 = (unsigned __int8)v10;
LABEL_45:
                v35 = v11;
                v25 = isalnum(v26);
                v11 = v35;
LABEL_46:
                if ( v10 == 95 || v25 )
                  continue;
                goto LABEL_30;
              }
              v13 = *(_DWORD *)(v7 + 76);
              if ( (v24 & 2) != 0 )
                goto LABEL_83;
            }
            break;
          }
          v42 = 128;
          v13 += *(_DWORD *)(v7 + 80);
LABEL_57:
          v12 = 131;
          if ( v13 > 0 )
            goto LABEL_15;
LABEL_41:
          if ( v10 == 128 )
            goto LABEL_30;
        }
      }
    }
    else
    {
      if ( v10 != 128 )
      {
        if ( v42 != 10 )
        {
LABEL_40:
          v12 = 0;
          goto LABEL_41;
        }
        v12 = *(_DWORD *)(v7 + 40) & 8;
        if ( !v12 )
          goto LABEL_41;
        goto LABEL_13;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        if ( v42 != 10 || (*(_BYTE *)(v7 + 40) & 8) == 0 )
          goto LABEL_30;
LABEL_13:
        v13 = *(_DWORD *)(v7 + 80);
        goto LABEL_14;
      }
    }
    v13 = *(_DWORD *)(v7 + 76);
    if ( v42 == 10 )
    {
      if ( (*(_BYTE *)(v7 + 40) & 8) != 0 )
      {
        v13 += *(_DWORD *)(v7 + 80);
        goto LABEL_57;
      }
      if ( v13 <= 0 )
        goto LABEL_20;
    }
    else if ( v13 <= 0 )
    {
      goto LABEL_19;
    }
    v12 = 129;
    goto LABEL_15;
  }
  v8 = a2;
  while ( 1 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(v7 + 8) + 8 * v6);
    if ( ((((unsigned int)v9 & 0xF8000000) - 1744830464LL) & 0xFFFFFFFFF0000000LL) != 0 )
      break;
LABEL_69:
    if ( a5 == ++v6 )
    {
LABEL_5:
      v43 = v8;
      goto LABEL_6;
    }
  }
  if ( (v9 & 0xF8000000) != 0x10000000 )
    goto LABEL_5;
  if ( a3 != v8 && *v8 == (_BYTE)v9 )
  {
    ++v8;
    goto LABEL_69;
  }
  return 0;
}
