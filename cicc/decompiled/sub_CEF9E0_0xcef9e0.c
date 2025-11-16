// Function: sub_CEF9E0
// Address: 0xcef9e0
//
__int64 __fastcall sub_CEF9E0(__int64 a1, const void *a2, size_t a3, unsigned int a4)
{
  size_t v6; // rdx
  char *v7; // r14
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  unsigned __int8 v14; // al
  __int64 *v15; // rdx
  const void *v16; // rax
  __int64 v17; // rdx
  unsigned __int8 v18; // al
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v23; // [rsp+8h] [rbp-48h]
  int v24; // [rsp+18h] [rbp-38h]
  int v25; // [rsp+1Ch] [rbp-34h]

  if ( !a1 )
    return 0;
  v6 = 0;
  v7 = off_4C5D0E0[0];
  if ( off_4C5D0E0[0] )
    v6 = strlen(off_4C5D0E0[0]);
  v8 = sub_BA8DC0(a1, (__int64)v7, v6);
  v9 = v8;
  if ( !v8 )
    return 0;
  v25 = sub_B91A00(v8);
  if ( !v25 )
    return 0;
  v10 = 0;
  v23 = a4;
  while ( 1 )
  {
    v11 = sub_B91A10(v9, v10);
    v12 = v11;
    if ( !v11 )
      goto LABEL_20;
    v13 = v11 - 16;
    v14 = *(_BYTE *)(v11 - 16);
    v15 = (v14 & 2) != 0 ? *(__int64 **)(v12 - 32) : (__int64 *)(v13 - 8LL * ((v14 >> 2) & 0xF));
    if ( *(_BYTE *)*v15 )
      goto LABEL_20;
    v16 = (const void *)sub_B91420(*v15);
    if ( v17 != a3 || a3 && memcmp(v16, a2, a3) )
      goto LABEL_20;
    v18 = *(_BYTE *)(v12 - 16);
    v19 = (v18 & 2) != 0 ? *(_QWORD *)(v12 - 32) : v13 - 8LL * ((v18 >> 2) & 0xF);
    v20 = *(_QWORD *)(v19 + 8);
    if ( *(_BYTE *)v20 != 1 )
      goto LABEL_20;
    v21 = *(_QWORD *)(v20 + 136);
    if ( *(_BYTE *)v21 != 17 )
      goto LABEL_20;
    if ( *(_DWORD *)(v21 + 32) > 0x40u )
      break;
    if ( v23 == *(_QWORD *)(v21 + 24) )
      return 1;
LABEL_20:
    if ( v25 == ++v10 )
      return 0;
  }
  v24 = *(_DWORD *)(v21 + 32);
  if ( v24 - (unsigned int)sub_C444A0(v21 + 24) > 0x40 || v23 != **(_QWORD **)(v21 + 24) )
    goto LABEL_20;
  return 1;
}
