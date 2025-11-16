// Function: sub_33D1E50
// Address: 0x33d1e50
//
char __fastcall sub_33D1E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char result; // al
  __int64 v7; // rdi
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned int v12; // edx
  unsigned int v13; // r8d
  const void *v14; // rax
  char v15; // [rsp-49h] [rbp-49h]
  char v16; // [rsp-49h] [rbp-49h]
  const void *v17; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v18; // [rsp-40h] [rbp-40h]
  const void *v19; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v20; // [rsp-30h] [rbp-30h]

  if ( *(_DWORD *)a1 != *(_DWORD *)(a2 + 24) )
    return 0;
  v7 = **(_QWORD **)(a2 + 40);
  v18 = 1;
  v17 = 0;
  if ( v7 && ((v8 = *(_DWORD *)(v7 + 24), v8 == 35) || v8 == 11) )
  {
    v11 = *(_QWORD *)(v7 + 96);
    v12 = *(_DWORD *)(v11 + 32);
    if ( v12 <= 0x40 )
    {
      v14 = *(const void **)(v11 + 24);
      v18 = v12;
      v17 = v14;
    }
    else
    {
      sub_C43990((__int64)&v17, v11 + 24);
      v12 = v18;
    }
  }
  else
  {
    result = sub_33D1410(v7, (__int64)&v17, a3, a4, a5);
    if ( !result )
    {
      if ( v18 <= 0x40 )
        return result;
      goto LABEL_8;
    }
    v12 = v18;
  }
  v13 = *(_DWORD *)(a1 + 16);
  if ( v12 != v13 )
  {
    if ( v12 >= v13 )
    {
      sub_C449B0((__int64)&v19, (const void **)(a1 + 8), v12);
      if ( v20 <= 0x40 )
      {
        result = v19 == v17;
        goto LABEL_23;
      }
      result = sub_C43C50((__int64)&v19, &v17);
    }
    else
    {
      sub_C449B0((__int64)&v19, &v17, v13);
      if ( *(_DWORD *)(a1 + 16) <= 0x40u )
        result = *(_QWORD *)(a1 + 8) == (_QWORD)v19;
      else
        result = sub_C43C50(a1 + 8, &v19);
      if ( v20 <= 0x40 )
        goto LABEL_23;
    }
    if ( v19 )
    {
      v16 = result;
      j_j___libc_free_0_0((unsigned __int64)v19);
      result = v16;
    }
LABEL_23:
    if ( v18 > 0x40 )
    {
LABEL_8:
      if ( v17 )
      {
        v15 = result;
        j_j___libc_free_0_0((unsigned __int64)v17);
        result = v15;
      }
      goto LABEL_10;
    }
    goto LABEL_10;
  }
  if ( v12 > 0x40 )
  {
    result = sub_C43C50(a1 + 8, &v17);
    goto LABEL_8;
  }
  result = *(_QWORD *)(a1 + 8) == (_QWORD)v17;
LABEL_10:
  if ( !result )
    return 0;
  v9 = *(_QWORD *)(a1 + 24);
  v10 = *(_QWORD *)(a2 + 40);
  if ( *(_QWORD *)(v10 + 40) != *(_QWORD *)v9 || *(_DWORD *)(v10 + 48) != *(_DWORD *)(v9 + 8) )
    return 0;
  if ( *(_BYTE *)(a1 + 36) )
    return (*(_DWORD *)(a1 + 32) & *(_DWORD *)(a2 + 28)) == *(_DWORD *)(a1 + 32);
  return result;
}
