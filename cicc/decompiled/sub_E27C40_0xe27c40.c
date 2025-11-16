// Function: sub_E27C40
// Address: 0xe27c40
//
unsigned __int64 __fastcall sub_E27C40(__int64 a1, unsigned __int64 *a2)
{
  char v4; // dl
  _QWORD *v5; // rax
  unsigned __int64 v6; // r14
  __int64 *v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // rbx
  char v13; // dl
  __int64 v14; // r14
  _QWORD *v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int16 v18; // ax
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rcx
  __int64 *v31; // rax
  __int64 *v32; // rbx
  __int64 v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // [rsp+0h] [rbp-50h]
  unsigned __int64 v37; // [rsp+8h] [rbp-48h]
  __int64 *v38; // [rsp+10h] [rbp-40h]
  _QWORD *v39; // [rsp+10h] [rbp-40h]
  unsigned __int64 v40; // [rsp+18h] [rbp-38h]

  ++a2[1];
  --*a2;
  v40 = sub_E219C0(a1, a2);
  if ( v40 && !v4 )
  {
    v5 = *(_QWORD **)(a1 + 16);
    v6 = (*v5 + v5[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v5[1] = v6 - *v5 + 32;
    v7 = *(__int64 **)(a1 + 16);
    v8 = v7[1];
    if ( v8 > v7[2] )
    {
      v20 = (__int64 *)sub_22077B0(32);
      v7 = v20;
      if ( v20 )
      {
        *v20 = 0;
        v20[1] = 0;
        v20[2] = 0;
        v20[3] = 0;
      }
      v21 = sub_2207820(4096);
      v7[2] = 4096;
      v37 = v21;
      v22 = v21;
      *v7 = v21;
      v23 = *(_QWORD *)(a1 + 16);
      v7[1] = 32;
      v7[3] = v23;
      *(_QWORD *)(a1 + 16) = v7;
      if ( !v22 )
      {
        v7[1] = 48;
        v36 = 32;
        goto LABEL_43;
      }
      *(_BYTE *)(v22 + 12) = 0;
      v9 = v22;
      *(_DWORD *)(v22 + 8) = 16;
      *(_QWORD *)(v22 + 16) = 0;
      *(_QWORD *)v22 = &unk_49E1158;
      v8 = 32;
      *(_QWORD *)(v22 + 24) = 0;
    }
    else if ( v6 )
    {
      *(_DWORD *)(v6 + 8) = 16;
      *(_BYTE *)(v6 + 12) = 0;
      *(_QWORD *)(v6 + 16) = 0;
      *(_QWORD *)v6 = &unk_49E1158;
      *(_QWORD *)(v6 + 24) = 0;
      v7 = *(__int64 **)(a1 + 16);
      v37 = v6;
      v9 = *v7;
      v8 = v7[1];
    }
    else
    {
      v37 = 0;
      v9 = *v7;
    }
    v10 = (v9 + v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v7[1] = v10 - v9 + 16;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v31 = (__int64 *)sub_22077B0(32);
      v32 = v31;
      if ( v31 )
      {
        *v31 = 0;
        v31[1] = 0;
        v31[2] = 0;
        v31[3] = 0;
      }
      v33 = sub_2207820(4096);
      v32[2] = 4096;
      v36 = v33;
      v34 = (_QWORD *)v33;
      *v32 = v33;
      v35 = *(_QWORD *)(a1 + 16);
      v32[1] = 16;
      v32[3] = v35;
      *(_QWORD *)(a1 + 16) = v32;
      if ( v34 )
      {
        *v34 = 0;
        v34[1] = 0;
      }
      goto LABEL_8;
    }
    v36 = 0;
    if ( !v10 )
    {
LABEL_8:
      v11 = (_QWORD *)v36;
      v12 = 0;
      while ( 1 )
      {
        v14 = sub_E219C0(a1, a2);
        if ( *(_BYTE *)(a1 + 8) || v13 )
          goto LABEL_19;
        v15 = *(_QWORD **)(a1 + 16);
        v16 = (*v15 + v15[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
        v15[1] = v16 + 32LL - *v15;
        if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
          break;
        v24 = (__int64 *)sub_22077B0(32);
        if ( v24 )
        {
          *v24 = 0;
          v24[1] = 0;
          v24[2] = 0;
          v24[3] = 0;
        }
        v38 = v24;
        v17 = sub_2207820(4096);
        v25 = *(_QWORD *)(a1 + 16);
        *v38 = v17;
        v38[3] = v25;
        v38[2] = 4096;
        *(_QWORD *)(a1 + 16) = v38;
        v38[1] = 32;
        if ( v17 )
          goto LABEL_14;
        *v11 = 0;
        if ( ++v12 >= v40 )
        {
LABEL_16:
          *(_QWORD *)(v37 + 16) = sub_E208B0((__int64 **)(a1 + 16), (_QWORD *)v36, v40);
          if ( !(unsigned __int8)sub_E20730(a2, 3u, "$$C")
            || (v18 = sub_E22E40(a1, (__int64 *)a2), *(_BYTE *)(v37 + 12) = v18, !HIBYTE(v18)) )
          {
            *(_QWORD *)(v37 + 24) = sub_E27700(a1, (__int64 *)a2, 0);
            return v37;
          }
          goto LABEL_19;
        }
LABEL_30:
        v26 = *(_QWORD **)(a1 + 16);
        v27 = (_QWORD *)((*v26 + v26[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
        v26[1] = (char *)v27 - *v26 + 16;
        if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
        {
          v29 = (_QWORD *)sub_22077B0(32);
          if ( v29 )
          {
            *v29 = 0;
            v29[1] = 0;
            v29[2] = 0;
            v29[3] = 0;
          }
          v39 = v29;
          v28 = (_QWORD *)sub_2207820(4096);
          v30 = *(_QWORD *)(a1 + 16);
          *v39 = v28;
          v39[3] = v30;
          v39[2] = 4096;
          *(_QWORD *)(a1 + 16) = v39;
          v39[1] = 16;
          if ( v28 )
          {
            *v28 = 0;
            v28[1] = 0;
          }
        }
        else
        {
          v28 = 0;
          if ( v27 )
          {
            *v27 = 0;
            v28 = v27;
            v27[1] = 0;
          }
        }
        v11[1] = v28;
        v11 = v28;
      }
      v17 = 0;
      if ( v16 )
      {
        v17 = v16;
LABEL_14:
        *(_DWORD *)(v17 + 8) = 23;
        *(_QWORD *)(v17 + 16) = v14;
        *(_BYTE *)(v17 + 24) = 0;
        *(_QWORD *)v17 = &unk_49E0F10;
      }
      *v11 = v17;
      if ( ++v12 >= v40 )
        goto LABEL_16;
      goto LABEL_30;
    }
    v36 = v10;
LABEL_43:
    *(_QWORD *)v36 = 0;
    *(_QWORD *)(v36 + 8) = 0;
    goto LABEL_8;
  }
LABEL_19:
  *(_BYTE *)(a1 + 8) = 1;
  return 0;
}
