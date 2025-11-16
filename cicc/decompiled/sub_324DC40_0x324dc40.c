// Function: sub_324DC40
// Address: 0x324dc40
//
__int64 __fastcall sub_324DC40(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 v4; // al
  __int64 v5; // rcx
  __int64 v6; // r15
  __int64 v7; // r13
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r10
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  const void *v18; // rcx
  __int64 v19; // rax
  size_t v20; // rdx
  size_t v21; // r8
  const char *v22; // r10
  unsigned __int8 v23; // al
  __int64 v24; // r14
  __int64 v25; // [rsp+8h] [rbp-38h]
  const char *v26; // [rsp+8h] [rbp-38h]
  const char *v27; // [rsp+8h] [rbp-38h]

  v2 = a2 - 16;
  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a2 - 32);
  else
    v5 = v2 - 8LL * ((v4 >> 2) & 0xF);
  v6 = (*(__int64 (__fastcall **)(__int64 *, _QWORD))(*a1 + 48))(a1, *(_QWORD *)(v5 + 8));
  v7 = (__int64)sub_3247C80((__int64)a1, (unsigned __int8 *)a2);
  if ( !v7 )
  {
    v7 = sub_324C6D0(a1, 57, v6, (unsigned __int8 *)a2);
    v9 = *(_BYTE *)(a2 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *(_QWORD *)(a2 - 32);
    else
      v10 = v2 - 8LL * ((v9 >> 2) & 0xF);
    v11 = *(_QWORD *)(v10 + 16);
    if ( v11 && (v12 = sub_B91420(v11), v14 = v13, v15 = v12, v13) )
    {
      v16 = *(_BYTE *)(a2 - 16);
      if ( (v16 & 2) != 0 )
        v17 = *(_QWORD *)(a2 - 32);
      else
        v17 = v2 - 8LL * ((v16 >> 2) & 0xF);
      v18 = *(const void **)(v17 + 16);
      if ( v18 )
      {
        v25 = v15;
        v19 = sub_B91420(*(_QWORD *)(v17 + 16));
        v15 = v25;
        v18 = (const void *)v19;
        v21 = v20;
      }
      else
      {
        v21 = 0;
      }
      v26 = (const char *)v15;
      sub_324AD70(a1, v7, 3, v18, v21);
      v22 = v26;
    }
    else
    {
      v22 = "(anonymous namespace)";
      v14 = 21;
    }
    v27 = v22;
    sub_32382C0(a1[26], (__int64)a1, *(_DWORD *)(a1[10] + 36), (__int64)v22, v14, v7);
    v23 = *(_BYTE *)(a2 - 16);
    if ( (v23 & 2) != 0 )
      v24 = *(_QWORD *)(a2 - 32);
    else
      v24 = v2 - 8LL * ((v23 >> 2) & 0xF);
    (*(void (__fastcall **)(__int64 *, const char *, __int64, __int64, _QWORD))(*a1 + 24))(
      a1,
      v27,
      v14,
      v7,
      *(_QWORD *)(v24 + 8));
    if ( *(char *)(a2 + 1) < 0 )
      sub_3249FA0(a1, v7, 137);
  }
  return v7;
}
