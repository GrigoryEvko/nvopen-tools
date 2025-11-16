// Function: sub_18430C0
// Address: 0x18430c0
//
__int64 __fastcall sub_18430C0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v7; // rbx
  _QWORD *v8; // r8
  int v9; // r9d
  unsigned __int8 v10; // al
  unsigned int v11; // r15d
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // rsi
  __int64 v16; // rbx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r14
  char v24; // dl
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // r15
  __int64 v31; // rbx
  unsigned int v32; // ebx
  __int64 v33; // rdx
  char v34; // al
  unsigned int v35; // ecx
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-40h]
  char v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]

  v7 = (unsigned __int64)sub_1648700(a2);
  v10 = *(_BYTE *)(v7 + 16);
  if ( v10 <= 0x17u )
    return 0;
  switch ( v10 )
  {
    case 0x19u:
      v13 = a4;
      v14 = a3;
      v15 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 56LL);
      if ( a4 != -1 )
        return sub_1842FA0(a1, v15, v13, v14, v8, v9);
      v32 = 0;
      v11 = 1;
      v33 = **(_QWORD **)(*(_QWORD *)(v15 + 24) + 16LL);
      v34 = *(_BYTE *)(v33 + 8);
      if ( v34 )
      {
        do
        {
          if ( v34 == 13 )
          {
            v35 = *(_DWORD *)(v33 + 12);
          }
          else
          {
            v35 = 1;
            if ( v34 == 14 )
              v35 = *(_DWORD *)(v33 + 32);
          }
          if ( v35 <= v32 )
            break;
          v36 = sub_1842FA0(a1, v15, v32, a3, v8, v9);
          if ( v11 )
            v11 = v36;
          ++v32;
          v33 = **(_QWORD **)(*(_QWORD *)(v15 + 24) + 16LL);
          v34 = *(_BYTE *)(v33 + 8);
        }
        while ( v34 );
        return v11;
      }
      return 1;
    case 0x57u:
      if ( (unsigned int)sub_1648720(a2) )
        a4 = **(_DWORD **)(v7 + 56);
      v31 = *(_QWORD *)(v7 + 8);
      if ( v31 )
      {
        while ( 1 )
        {
          v11 = sub_18430C0(a1, v31, a3, a4);
          if ( !v11 )
            break;
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            return v11;
        }
        return 0;
      }
      return 1;
    case 0x4Eu:
      v16 = v7 | 4;
      break;
    case 0x1Du:
      v16 = v7 & 0xFFFFFFFFFFFFFFFBLL;
      break;
    default:
      return 0;
  }
  v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v41 = (v16 >> 2) & 1;
  if ( v41 )
  {
    v18 = *(_QWORD *)(v17 - 24);
    if ( *(_BYTE *)(v18 + 16) )
      return 0;
  }
  else
  {
    v18 = *(_QWORD *)(v17 - 72);
    if ( *(_BYTE *)(v18 + 16) )
      return 0;
  }
  if ( *(char *)(v17 + 23) >= 0 )
    goto LABEL_25;
  v40 = v17;
  v19 = sub_1648A40(v17);
  v17 = v40;
  v21 = v19 + v20;
  if ( *(char *)(v40 + 23) >= 0 )
  {
    v23 = v21 >> 4;
  }
  else
  {
    v22 = sub_1648A40(v40);
    v17 = v40;
    v23 = (v21 - v22) >> 4;
  }
  if ( !(_DWORD)v23 )
    goto LABEL_25;
  v24 = *(_BYTE *)(v17 + 23);
  if ( (v24 & 0x40) != 0 )
    v25 = *(_QWORD *)(v17 - 8);
  else
    v25 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
  v26 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2 - v25) >> 3);
  if ( v41 )
  {
    if ( v24 >= 0 )
      JUMPOUT(0x41B404);
    v42 = v17;
    v27 = sub_1648A40(v17);
    v17 = v42;
    if ( *(_DWORD *)(v27 + 8) <= (unsigned int)v26 )
    {
      if ( *(char *)(v42 + 23) < 0 )
      {
        v28 = sub_1648A40(v42);
        v17 = v42;
        if ( *(_DWORD *)(v28 + v29 - 4) > (unsigned int)v26 )
          return 0;
        goto LABEL_25;
      }
LABEL_54:
      BUG();
    }
  }
  else
  {
    if ( v24 >= 0 )
      BUG();
    v43 = v17;
    v37 = sub_1648A40(v17);
    v17 = v43;
    if ( *(_DWORD *)(v37 + 8) <= (unsigned int)v26 )
    {
      if ( *(char *)(v43 + 23) >= 0 )
        goto LABEL_54;
      v38 = sub_1648A40(v43);
      if ( *(_DWORD *)(v38 + v39 - 4) > (unsigned int)v26 )
        return 0;
      v17 = v43;
    }
  }
LABEL_25:
  v30 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2 - (v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF))) >> 3);
  if ( *(_DWORD *)(*(_QWORD *)(v18 + 24) + 12LL) - 1 <= (unsigned int)v30 )
    return 0;
  v14 = a3;
  v15 = v18;
  v13 = (unsigned int)v30 | 0x100000000LL;
  return sub_1842FA0(a1, v15, v13, v14, v8, v9);
}
