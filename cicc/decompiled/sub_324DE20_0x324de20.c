// Function: sub_324DE20
// Address: 0x324de20
//
__int64 __fastcall sub_324DE20(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 v5; // al
  __int64 v6; // rcx
  __int64 v7; // r15
  __int64 v8; // r13
  unsigned __int8 v10; // al
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rdx
  const void *v14; // rcx
  size_t v15; // rdx
  size_t v16; // r8
  void (__fastcall *v17)(__int64 *, __int64, __int64, __int64, __int64); // r15
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rdx
  const void *v28; // rcx
  size_t v29; // rdx
  size_t v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  const void *v34; // rcx
  size_t v35; // rdx
  size_t v36; // r8
  __int64 v37; // rdi
  __int64 v38; // rdx
  unsigned __int8 v39; // al
  __int64 v40; // rdx
  const void *v41; // rcx
  size_t v42; // rdx
  size_t v43; // r8
  __int64 v44; // rsi
  unsigned __int8 v45; // al
  __int64 *v46; // r14
  unsigned int v47; // eax
  __int64 v48; // r8
  __int64 v49; // [rsp+8h] [rbp-48h]
  int v50; // [rsp+1Ch] [rbp-34h]

  v2 = a2 - 16;
  v5 = *(_BYTE *)(a2 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_QWORD *)(a2 - 32);
  else
    v6 = v2 - 8LL * ((v5 >> 2) & 0xF);
  v7 = (*(__int64 (__fastcall **)(__int64 *, _QWORD))(*a1 + 48))(a1, *(_QWORD *)(v6 + 8));
  v8 = (__int64)sub_3247C80((__int64)a1, (unsigned __int8 *)a2);
  if ( v8 )
    return v8;
  v8 = sub_324C6D0(a1, 30, v7, (unsigned __int8 *)a2);
  v10 = *(_BYTE *)(a2 - 16);
  if ( (v10 & 2) != 0 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( !v11 )
      goto LABEL_30;
  }
  else
  {
    v11 = *(_QWORD *)(a2 - 8LL * ((v10 >> 2) & 0xF));
    if ( !v11 )
      goto LABEL_19;
  }
  sub_B91420(v11);
  v10 = *(_BYTE *)(a2 - 16);
  if ( v12 )
  {
    if ( (v10 & 2) != 0 )
      v13 = *(_QWORD *)(a2 - 32);
    else
      v13 = v2 - 8LL * ((v10 >> 2) & 0xF);
    v14 = *(const void **)(v13 + 16);
    if ( v14 )
    {
      v14 = (const void *)sub_B91420(*(_QWORD *)(v13 + 16));
      v16 = v15;
    }
    else
    {
      v16 = 0;
    }
    sub_324AD70(a1, v8, 3, v14, v16);
    v17 = *(void (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*a1 + 24);
    v18 = *(_BYTE *)(a2 - 16);
    if ( (v18 & 2) != 0 )
      v19 = *(_QWORD *)(a2 - 32);
    else
      v19 = v2 - 8LL * ((v18 >> 2) & 0xF);
    v20 = *(_QWORD *)(v19 + 8);
    v21 = *(_QWORD *)(v19 + 16);
    if ( v21 )
    {
      v49 = *(_QWORD *)(v19 + 8);
      v22 = sub_B91420(*(_QWORD *)(v19 + 16));
      v20 = v49;
      v21 = v22;
    }
    else
    {
      v23 = 0;
    }
    v17(a1, v21, v23, v8, v20);
    v10 = *(_BYTE *)(a2 - 16);
  }
  if ( (v10 & 2) == 0 )
  {
LABEL_19:
    v24 = *(_QWORD *)(v2 - 8LL * ((v10 >> 2) & 0xF) + 24);
    if ( !v24 )
      goto LABEL_20;
    goto LABEL_31;
  }
LABEL_30:
  v24 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
  if ( !v24 )
    goto LABEL_38;
LABEL_31:
  sub_B91420(v24);
  v10 = *(_BYTE *)(a2 - 16);
  if ( v32 )
  {
    if ( (v10 & 2) != 0 )
      v33 = *(_QWORD *)(a2 - 32);
    else
      v33 = v2 - 8LL * ((v10 >> 2) & 0xF);
    v34 = *(const void **)(v33 + 24);
    if ( v34 )
    {
      v34 = (const void *)sub_B91420(*(_QWORD *)(v33 + 24));
      v36 = v35;
    }
    else
    {
      v36 = 0;
    }
    sub_324AD70(a1, v8, 15873, v34, v36);
    v10 = *(_BYTE *)(a2 - 16);
  }
  if ( (v10 & 2) != 0 )
  {
LABEL_38:
    v25 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
    if ( !v25 )
      goto LABEL_39;
    goto LABEL_21;
  }
LABEL_20:
  v25 = *(_QWORD *)(v2 - 8LL * ((v10 >> 2) & 0xF) + 32);
  if ( !v25 )
  {
LABEL_28:
    v31 = v2 - 8LL * ((v10 >> 2) & 0xF);
    goto LABEL_40;
  }
LABEL_21:
  sub_B91420(v25);
  v10 = *(_BYTE *)(a2 - 16);
  if ( v26 )
  {
    if ( (v10 & 2) != 0 )
      v27 = *(_QWORD *)(a2 - 32);
    else
      v27 = v2 - 8LL * ((v10 >> 2) & 0xF);
    v28 = *(const void **)(v27 + 32);
    if ( v28 )
    {
      v28 = (const void *)sub_B91420(*(_QWORD *)(v27 + 32));
      v30 = v29;
    }
    else
    {
      v30 = 0;
    }
    sub_324AD70(a1, v8, 15872, v28, v30);
    v10 = *(_BYTE *)(a2 - 16);
  }
  if ( (v10 & 2) == 0 )
    goto LABEL_28;
LABEL_39:
  v31 = *(_QWORD *)(a2 - 32);
LABEL_40:
  v37 = *(_QWORD *)(v31 + 40);
  if ( v37 )
  {
    sub_B91420(v37);
    if ( v38 )
    {
      v39 = *(_BYTE *)(a2 - 16);
      if ( (v39 & 2) != 0 )
        v40 = *(_QWORD *)(a2 - 32);
      else
        v40 = v2 - 8LL * ((v39 >> 2) & 0xF);
      v41 = *(const void **)(v40 + 40);
      if ( v41 )
      {
        v41 = (const void *)sub_B91420(*(_QWORD *)(v40 + 40));
        v43 = v42;
      }
      else
      {
        v43 = 0;
      }
      sub_324AD70(a1, v8, 15879, v41, v43);
    }
  }
  v44 = a2;
  if ( *(_BYTE *)a2 == 16
    || ((v45 = *(_BYTE *)(a2 - 16), (v45 & 2) == 0)
      ? (v46 = (__int64 *)(v2 - 8LL * ((v45 >> 2) & 0xF)))
      : (v46 = *(__int64 **)(a2 - 32)),
        (v44 = *v46) != 0) )
  {
    v47 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 80))(a1, v44);
    BYTE2(v50) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(v8 + 8), 58, v50, v47);
  }
  v48 = *(unsigned int *)(a2 + 4);
  if ( (_DWORD)v48 )
  {
    BYTE2(v50) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(v8 + 8), 59, v50, v48);
  }
  if ( *(char *)(a2 + 1) < 0 )
    sub_3249FA0(a1, v8, 60);
  return v8;
}
