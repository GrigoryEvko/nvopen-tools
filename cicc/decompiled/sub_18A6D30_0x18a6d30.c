// Function: sub_18A6D30
// Address: 0x18a6d30
//
void __fastcall sub_18A6D30(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r14
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // ecx
  _QWORD *v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // ecx
  _QWORD *v20; // r8
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  _QWORD *v30; // [rsp+18h] [rbp-48h]
  _QWORD *i; // [rsp+18h] [rbp-48h]
  _QWORD *v32; // [rsp+20h] [rbp-40h]
  _QWORD *v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+28h] [rbp-38h]
  __int64 v35; // [rsp+28h] [rbp-38h]

  v1 = a1;
  v2 = (_QWORD *)*a1;
  v3 = (_QWORD *)*(a1 - 1);
  v4 = *(_QWORD *)(*a1 + 120LL);
  if ( *(_QWORD *)(*a1 + 72LL) )
    goto LABEL_2;
LABEL_31:
  if ( v4 )
  {
    v6 = v2[13];
    goto LABEL_5;
  }
LABEL_15:
  while ( 1 )
  {
    v16 = v3[15];
    if ( !v3[9] )
      break;
    v17 = v3[7];
    if ( v16 )
    {
      v18 = v3[13];
      v19 = *(_DWORD *)(v18 + 32);
      if ( *(_DWORD *)(v17 + 32) >= v19
        && (*(_DWORD *)(v17 + 32) != v19 || *(_DWORD *)(v17 + 36) >= *(_DWORD *)(v18 + 36)) )
      {
        goto LABEL_19;
      }
    }
    if ( *(_QWORD *)(v17 + 40) >= v4 )
      goto LABEL_39;
LABEL_30:
    *v1-- = v3;
    v4 = v2[15];
    v3 = (_QWORD *)*(v1 - 1);
    if ( !v2[9] )
      goto LABEL_31;
LABEL_2:
    v5 = v2[7];
    if ( v4 )
    {
      v6 = v2[13];
      v7 = *(_DWORD *)(v6 + 32);
      if ( *(_DWORD *)(v5 + 32) >= v7 && (*(_DWORD *)(v5 + 32) != v7 || *(_DWORD *)(v5 + 36) >= *(_DWORD *)(v6 + 36)) )
      {
LABEL_5:
        v8 = *(_QWORD **)(v6 + 64);
        v4 = 0;
        v30 = (_QWORD *)(v6 + 48);
        if ( v8 == (_QWORD *)(v6 + 48) )
          continue;
        while ( 1 )
        {
          v9 = v8[23];
          if ( v8[17] )
          {
            v10 = v8[15];
            if ( v9 )
            {
              v11 = v8[21];
              v12 = *(_DWORD *)(v11 + 32);
              if ( *(_DWORD *)(v10 + 32) >= v12
                && (*(_DWORD *)(v10 + 32) != v12 || *(_DWORD *)(v10 + 36) >= *(_DWORD *)(v11 + 36)) )
              {
                goto LABEL_10;
              }
            }
            v4 += *(_QWORD *)(v10 + 40);
          }
          else if ( v9 )
          {
            v11 = v8[21];
LABEL_10:
            v13 = *(_QWORD *)(v11 + 64);
            v28 = v11 + 48;
            if ( v13 != v11 + 48 )
            {
              v14 = 0;
              do
              {
                v32 = v8;
                v34 = v13;
                v14 += sub_18A58D0(v13 + 64);
                v15 = sub_220EF30(v34);
                v8 = v32;
                v13 = v15;
              }
              while ( v28 != v15 );
              v4 += v14;
            }
          }
          v8 = (_QWORD *)sub_220EF30(v8);
          if ( v30 == v8 )
            goto LABEL_15;
        }
      }
    }
    v4 = *(_QWORD *)(v5 + 40);
  }
  if ( v16 )
  {
    v18 = v3[13];
LABEL_19:
    v20 = *(_QWORD **)(v18 + 64);
    v16 = 0;
    for ( i = (_QWORD *)(v18 + 48); i != v20; v20 = (_QWORD *)sub_220EF30(v20) )
    {
      v21 = v20[23];
      if ( v20[17] )
      {
        v22 = v20[15];
        if ( v21 )
        {
          v23 = v20[21];
          v24 = *(_DWORD *)(v23 + 32);
          if ( *(_DWORD *)(v22 + 32) >= v24
            && (*(_DWORD *)(v22 + 32) != v24 || *(_DWORD *)(v22 + 36) >= *(_DWORD *)(v23 + 36)) )
          {
            goto LABEL_24;
          }
        }
        v16 += *(_QWORD *)(v22 + 40);
      }
      else if ( v21 )
      {
        v23 = v20[21];
LABEL_24:
        v25 = *(_QWORD *)(v23 + 64);
        v27 = v23 + 48;
        if ( v25 != v23 + 48 )
        {
          v29 = 0;
          do
          {
            v33 = v20;
            v35 = v25;
            v29 += sub_18A58D0(v25 + 64);
            v26 = sub_220EF30(v35);
            v20 = v33;
            v25 = v26;
          }
          while ( v27 != v26 );
          v16 += v29;
        }
      }
    }
  }
  if ( v16 < v4 )
    goto LABEL_30;
LABEL_39:
  *v1 = v2;
}
