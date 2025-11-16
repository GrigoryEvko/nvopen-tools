// Function: sub_109D820
// Address: 0x109d820
//
bool __fastcall sub_109D820(_BYTE *a1, _QWORD *a2, __int64 a3, _BYTE *a4)
{
  char v7; // al
  bool result; // al
  __int64 v9; // rax
  __int64 v10; // rdi
  _BYTE *v11; // r12
  unsigned int v12; // eax
  unsigned __int64 v13; // r15
  bool v14; // cc
  unsigned int v15; // ebx
  const void *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdx
  _BYTE *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // rdx
  _BYTE *v27; // rax
  bool v28; // [rsp+8h] [rbp-58h]
  _BYTE *v29; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-48h]
  const void *v32; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-38h]

  *a4 = 0;
  v7 = *a1;
  if ( *a1 == 52 )
  {
    v17 = *((_QWORD *)a1 - 8);
    if ( !v17 )
      return 0;
    *a2 = v17;
    v18 = *((_QWORD *)a1 - 4);
    v19 = v18 + 24;
    if ( *(_BYTE *)v18 != 17 )
    {
      v29 = a4;
      v22 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17;
      if ( (unsigned int)v22 > 1 || *(_BYTE *)v18 > 0x15u || (v23 = sub_AD7630(v18, 0, v22)) == 0 || *v23 != 17 )
      {
        v7 = *a1;
        goto LABEL_2;
      }
      a4 = v29;
      v19 = (__int64)(v23 + 24);
    }
    *a4 = 1;
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
      goto LABEL_31;
    goto LABEL_30;
  }
LABEL_2:
  if ( v7 == 51 )
  {
    v20 = *((_QWORD *)a1 - 8);
    if ( !v20 )
      return 0;
    *a2 = v20;
    v21 = *((_QWORD *)a1 - 4);
    v19 = v21 + 24;
    if ( *(_BYTE *)v21 != 17 )
    {
      v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v21 + 8) + 8LL) - 17;
      if ( (unsigned int)v24 > 1
        || *(_BYTE *)v21 > 0x15u
        || (v25 = sub_AD7630(v21, 0, v24)) == 0
        || (v19 = (__int64)(v25 + 24), *v25 != 17) )
      {
        v7 = *a1;
        goto LABEL_3;
      }
    }
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
LABEL_31:
      sub_C43990(a3, v19);
      return 1;
    }
LABEL_30:
    if ( *(_DWORD *)(v19 + 8) <= 0x40u )
    {
      *(_QWORD *)a3 = *(_QWORD *)v19;
      *(_DWORD *)(a3 + 8) = *(_DWORD *)(v19 + 8);
      return 1;
    }
    goto LABEL_31;
  }
LABEL_3:
  if ( v7 != 57 )
    return 0;
  v9 = *((_QWORD *)a1 - 8);
  if ( !v9 )
    return 0;
  *a2 = v9;
  v10 = *((_QWORD *)a1 - 4);
  v11 = (_BYTE *)(v10 + 24);
  if ( *(_BYTE *)v10 != 17 )
  {
    v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
    if ( (unsigned int)v26 > 1 )
      return 0;
    if ( *(_BYTE *)v10 > 0x15u )
      return 0;
    v27 = sub_AD7630(v10, 0, v26);
    if ( !v27 || *v27 != 17 )
      return 0;
    v11 = v27 + 24;
  }
  v31 = *((_DWORD *)v11 + 2);
  if ( v31 > 0x40 )
    sub_C43780((__int64)&v30, (const void **)v11);
  else
    v30 = *(_QWORD *)v11;
  sub_C46A40((__int64)&v30, 1);
  v12 = v31;
  v13 = v30;
  v31 = 0;
  v33 = v12;
  v32 = (const void *)v30;
  if ( v12 <= 0x40 )
  {
    if ( v30 && (v30 & (v30 - 1)) == 0 )
    {
LABEL_16:
      v33 = *((_DWORD *)v11 + 2);
      if ( v33 > 0x40 )
        sub_C43780((__int64)&v32, (const void **)v11);
      else
        v32 = *(const void **)v11;
      sub_C46A40((__int64)&v32, 1);
      v14 = *(_DWORD *)(a3 + 8) <= 0x40u;
      v15 = v33;
      v33 = 0;
      v16 = v32;
      if ( v14 || !*(_QWORD *)a3 )
      {
        *(_QWORD *)a3 = v32;
        *(_DWORD *)(a3 + 8) = v15;
      }
      else
      {
        j_j___libc_free_0_0(*(_QWORD *)a3);
        v14 = v33 <= 0x40;
        *(_QWORD *)a3 = v16;
        *(_DWORD *)(a3 + 8) = v15;
        if ( !v14 )
        {
          if ( v32 )
            j_j___libc_free_0_0(v32);
        }
      }
      return 1;
    }
    return 0;
  }
  result = (unsigned int)sub_C44630((__int64)&v32) == 1;
  if ( v13 )
  {
    v28 = result;
    j_j___libc_free_0_0(v13);
    result = v28;
    if ( v31 > 0x40 )
    {
      if ( v30 )
      {
        j_j___libc_free_0_0(v30);
        result = v28;
      }
    }
  }
  if ( result )
    goto LABEL_16;
  return result;
}
