// Function: sub_5D0FF0
// Address: 0x5d0ff0
//
void sub_5D0FF0()
{
  _QWORD *v0; // r13
  _BOOL4 v1; // eax
  __int64 v2; // rsi
  _BOOL4 v3; // r9d
  char v4; // al
  _BOOL4 v5; // eax
  _QWORD *v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // r14
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  char *v13; // r12
  size_t v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r12
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  char v26; // dl
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  _QWORD *v35; // r14
  __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 v37; // [rsp+0h] [rbp-80h]
  _OWORD v38[7]; // [rsp+10h] [rbp-70h] BYREF

  v0 = (_QWORD *)qword_4CF6E30;
  qword_4CF6E28 = 0;
  qword_4CF6E30 = 0;
  if ( v0 )
  {
    while ( 1 )
    {
      v6 = v0;
      v0 = (_QWORD *)*v0;
      v7 = v6[1];
      v8 = v6 + 4;
      if ( v7 )
      {
        v8 = (_QWORD *)(v7 + 48);
        if ( (*(_BYTE *)(v7 + 81) & 2) != 0 )
        {
          if ( *(_BYTE *)(v7 + 80) != 7
            || (v9 = *(_QWORD *)(v7 + 88), *(_BYTE *)(v9 + 136) != 2)
            || *(_BYTE *)(v9 + 177) )
          {
            sub_6851C0(1154, v8);
          }
        }
      }
      v10 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v11 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v12 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v38[0] = _mm_loadu_si128(xmmword_4F06660);
      v38[1] = v10;
      v38[2] = v11;
      v38[3] = v12;
      v13 = (char *)v6[3];
      *((_QWORD *)&v38[0] + 1) = *v8;
      v14 = strlen(v13);
      sub_878540(v13, v14);
      v15 = sub_7D5DD0(v38, 32);
      v16 = v6[1];
      v17 = v15;
      if ( !v16 )
        break;
      v36 = v6[1];
      v1 = sub_5C64B0(v15, v16);
      v2 = v36;
      if ( !v1 )
      {
        v21 = (__int64 *)sub_881B20(qword_4CF6E18, v6[3], 0);
        if ( v21 && (v22 = *v21) != 0 )
        {
          v2 = v6[1];
          v17 = v22;
          if ( !v2 )
          {
            v18 = *(_BYTE *)(v22 + 80);
            if ( v18 == 11 )
              goto LABEL_35;
LABEL_20:
            if ( v18 == 17 )
              goto LABEL_28;
            if ( v18 == 7 )
            {
              *(_QWORD *)(*(_QWORD *)(v17 + 88) + 144LL) = v6[2];
              v19 = *(_QWORD *)(v17 + 88);
              goto LABEL_23;
            }
            goto LABEL_10;
          }
        }
        else
        {
          v2 = v6[1];
          if ( !v2 )
            break;
        }
      }
      v3 = sub_5C64B0(v17, v2);
      v4 = *(_BYTE *)(v2 + 80);
      if ( v3 )
      {
        if ( *(_BYTE *)(v17 + 80) == v4 )
        {
          if ( v4 == 11 )
          {
            v29 = *(_QWORD *)(v2 + 88);
            v30 = *(_QWORD *)(v17 + 88);
            if ( ((*(_BYTE *)(v30 - 8) ^ *(_BYTE *)(v29 - 8)) & 2) != 0 )
              goto LABEL_63;
            if ( (*(_BYTE *)(v29 + 201) & 1) != 0 )
            {
              v37 = *(_QWORD *)(v30 + 152);
              v35 = *(_QWORD **)(v37 + 168);
              if ( !(unsigned int)sub_8D2340(*(_QWORD *)(v37 + 160)) || *v35 )
                sub_686870(2536, v6 + 4, v17, v37);
            }
            v31 = *(_QWORD *)(v6[1] + 88LL);
            v32 = *(_QWORD *)(v31 + 256);
            if ( !v32 )
              v32 = sub_726210(v31);
            *(_QWORD *)(v32 + 8) = *(_QWORD *)(v17 + 88);
            sub_5C67D0((__int64)v6);
          }
          else
          {
            if ( v4 != 7 )
              sub_721090(v17);
            v33 = *(_QWORD *)(v2 + 88);
            v34 = *(_QWORD *)(v17 + 88);
            if ( ((*(_BYTE *)(v34 - 8) ^ *(_BYTE *)(v33 - 8)) & 2) != 0 )
            {
LABEL_63:
              if ( qword_4CF6E30 )
                *(_QWORD *)qword_4CF6E28 = v6;
              else
                qword_4CF6E30 = (__int64)v6;
              qword_4CF6E28 = (__int64)v6;
              *v6 = 0;
              goto LABEL_11;
            }
            *(_QWORD *)(v33 + 232) = v34;
            sub_5C67D0((__int64)v6);
          }
          sub_5D0F70(v6[1]);
          sub_8767A0(12, v17, v6[1] + 48LL, 1);
        }
        else
        {
          sub_6854C0(1153, v2 + 48, v17);
        }
      }
      else
      {
        if ( v4 == 7 )
        {
          v5 = (*(_BYTE *)(*(_QWORD *)(v2 + 88) + 168LL) & 0x10) != 0;
        }
        else
        {
          if ( v4 != 11 )
            goto LABEL_9;
          v5 = (*(_BYTE *)(*(_QWORD *)(v2 + 88) + 200LL) & 0x40) != 0;
        }
        if ( !v5 )
LABEL_9:
          sub_6849F0(qword_4F077A8 < 0x9C40u ? 5 : 7, 1152, v6 + 4, v6[3]);
      }
LABEL_10:
      *v6 = qword_4CF6E20;
      qword_4CF6E20 = (__int64)v6;
LABEL_11:
      if ( !v0 )
        return;
    }
    if ( !v17 )
    {
      v25 = *(_QWORD *)(*(_QWORD *)&v38[0] + 40LL);
      if ( v25 )
      {
        while ( 1 )
        {
          v26 = *(_BYTE *)(v25 + 80);
          if ( v26 == 15 )
            break;
          if ( v26 == 14 )
          {
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 88) + 8LL) + 144LL) = v6[2];
            goto LABEL_10;
          }
          v25 = *(_QWORD *)(v25 + 8);
          if ( !v25 )
            goto LABEL_10;
        }
        v27 = *(_QWORD *)(*(_QWORD *)(v25 + 88) + 8LL);
        v28 = *(_QWORD *)(v27 + 256);
        if ( !v28 )
          v28 = sub_726210(v27);
        *(_QWORD *)(v28 + 40) = v6[2];
      }
      goto LABEL_10;
    }
    v18 = *(_BYTE *)(v17 + 80);
    if ( v18 == 11 )
    {
LABEL_35:
      v23 = *(_QWORD *)(v17 + 88);
      v24 = *(_QWORD *)(v23 + 256);
      if ( !v24 )
        v24 = sub_726210(v23);
      *(_QWORD *)(v24 + 40) = v6[2];
      v19 = *(_QWORD *)(v17 + 88);
LABEL_23:
      if ( !v19 )
        goto LABEL_10;
      if ( (*(_BYTE *)(v19 + 88) & 0x70) == 0x30 )
        goto LABEL_10;
      if ( *(_BYTE *)(v17 + 80) == 7 )
      {
        v20 = *(_QWORD *)(v19 + 40);
        if ( !v20 || *(_BYTE *)(v20 + 28) != 3 )
          goto LABEL_10;
      }
LABEL_28:
      sub_6854C0(1199, v8, v17);
      goto LABEL_10;
    }
    goto LABEL_20;
  }
}
