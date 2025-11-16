// Function: sub_373FD70
// Address: 0x373fd70
//
__int64 __fastcall sub_373FD70(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // r15
  unsigned __int8 v9; // al
  unsigned __int8 **v10; // rdx
  unsigned __int8 *v11; // rax
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdx
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int64 v18; // rdi
  size_t v19; // rdx
  size_t v20; // r11
  const char *v21; // r10
  unsigned __int8 v22; // al
  unsigned __int8 **v23; // rdx
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int8 v27; // al
  __int64 v28; // rdx
  char *v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v7 = (__int64)sub_3247C80((__int64)a1, (unsigned __int8 *)a2);
  if ( v7 )
    return v7;
  v31 = a2 - 16;
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(unsigned __int8 ***)(a2 - 32);
  else
    v10 = (unsigned __int8 **)(v31 - 8LL * ((v9 >> 2) & 0xF));
  v11 = sub_373FC60(a1, *v10);
  v7 = sub_324C6D0(a1, 26, (__int64)v11, (unsigned __int8 *)a2);
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = v31 - 8LL * ((v12 >> 2) & 0xF);
  v14 = *(_QWORD *)(v13 + 16);
  if ( v14 && (sub_B91420(v14), v15) )
  {
    v16 = *(_BYTE *)(a2 - 16);
    if ( (v16 & 2) != 0 )
      v17 = *(_QWORD *)(a2 - 32);
    else
      v17 = v31 - 8LL * ((v16 >> 2) & 0xF);
    v18 = *(_QWORD *)(v17 + 16);
    if ( v18 )
    {
      v18 = sub_B91420(v18);
      v20 = v19;
    }
    else
    {
      v20 = 0;
    }
    v21 = (const char *)v18;
  }
  else
  {
    v21 = "_BLNK_";
    v20 = 6;
  }
  v29 = (char *)v21;
  v30 = v20;
  sub_324AD70(a1, v7, 3, v21, v20);
  v22 = *(_BYTE *)(a2 - 16);
  if ( (v22 & 2) != 0 )
    v23 = *(unsigned __int8 ***)(a2 - 32);
  else
    v23 = (unsigned __int8 **)(v31 - 8LL * ((v22 >> 2) & 0xF));
  sub_3736650((__int64)a1, v29, v30, v7, *v23);
  v24 = *(_BYTE *)(a2 - 16);
  if ( (v24 & 2) != 0 )
  {
    v25 = *(_QWORD *)(a2 - 32);
    v26 = *(_QWORD *)(v25 + 24);
    if ( !v26 )
      goto LABEL_22;
  }
  else
  {
    v25 = v31 - 8LL * ((v24 >> 2) & 0xF);
    v26 = *(_QWORD *)(v25 + 24);
    if ( !v26 )
      goto LABEL_22;
  }
  sub_3249CA0(a1, v7, *(_DWORD *)(a2 + 4), v26);
  v27 = *(_BYTE *)(a2 - 16);
  if ( (v27 & 2) != 0 )
    v25 = *(_QWORD *)(a2 - 32);
  else
    v25 = v31 - 8LL * ((v27 >> 2) & 0xF);
LABEL_22:
  v28 = *(_QWORD *)(v25 + 8);
  if ( v28 )
    sub_3739A60(a1, v7, v28, a3, a4);
  return v7;
}
