// Function: sub_D93590
// Address: 0xd93590
//
__int64 __fastcall sub_D93590(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, unsigned int a4)
{
  unsigned __int8 v6; // al
  unsigned __int8 v7; // dl
  __int64 result; // rax
  int v9; // eax
  int v10; // edx
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // esi
  __int64 v14; // r8
  int v15; // esi
  unsigned int v16; // r9d
  __int64 *v17; // rax
  __int64 v18; // r10
  _QWORD **v19; // rax
  _QWORD *v20; // rdx
  unsigned int v21; // r9d
  __int64 *v22; // rdx
  __int64 v23; // r10
  _QWORD **v24; // rdx
  _QWORD *v25; // rdx
  int i; // esi
  unsigned int v27; // ebx
  __int64 v28; // r15
  unsigned __int8 *v29; // rdx
  unsigned __int8 *v30; // rcx
  const char *v31; // rax
  size_t v32; // rdx
  size_t v33; // rbx
  const char *v34; // r13
  const char *v35; // rax
  size_t v36; // rdx
  size_t v37; // r12
  int v38; // eax
  int v39; // eax
  __int64 v40; // r13
  int v41; // eax
  int v42; // edx
  int v43; // r11d
  int v44; // r11d

  if ( a4 > (unsigned int)qword_4F893E8 )
    return 0;
  v6 = *(_BYTE *)(*((_QWORD *)a2 + 1) + 8LL) == 14;
  v7 = *(_BYTE *)(*((_QWORD *)a3 + 1) + 8LL) == 14;
  if ( v6 != v7 )
    return v6 - (unsigned int)v7;
  v9 = *a2;
  v10 = *a3;
  if ( (_BYTE)v9 != (_BYTE)v10 )
    return (unsigned int)(v9 - v10);
  if ( (_BYTE)v9 == 22 )
    return (unsigned int)(*((_DWORD *)a2 + 8) - *((_DWORD *)a3 + 8));
  if ( (unsigned __int8)v9 <= 3u && (a2[32] & 0xFu) - 7 > 1 && (a3[32] & 0xFu) - 7 > 1 )
  {
    v31 = sub_BD5D20((__int64)a2);
    v33 = v32;
    v34 = v31;
    v35 = sub_BD5D20((__int64)a3);
    v37 = v36;
    if ( v33 <= v36 )
      v36 = v33;
    if ( v36 )
    {
      v38 = memcmp(v34, v35, v36);
      if ( v38 )
        return (v38 >> 31) | 1u;
    }
    result = 0;
    if ( v33 != v37 )
      return v33 < v37 ? -1 : 1;
    return result;
  }
  if ( (unsigned __int8)v9 <= 0x1Cu )
    return 0;
  v11 = *((_QWORD *)a2 + 5);
  v12 = *((_QWORD *)a3 + 5);
  if ( v11 != v12 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    v14 = *(_QWORD *)(a1 + 8);
    if ( v13 )
    {
      v15 = v13 - 1;
      v16 = v15 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v17 = (__int64 *)(v14 + 16LL * v16);
      v18 = *v17;
      if ( v11 == *v17 )
      {
LABEL_16:
        v19 = (_QWORD **)v17[1];
        if ( v19 )
        {
          v20 = *v19;
          for ( LODWORD(v19) = 1; v20; LODWORD(v19) = (_DWORD)v19 + 1 )
            v20 = (_QWORD *)*v20;
        }
      }
      else
      {
        v41 = 1;
        while ( v18 != -4096 )
        {
          v44 = v41 + 1;
          v16 = v15 & (v41 + v16);
          v17 = (__int64 *)(v14 + 16LL * v16);
          v18 = *v17;
          if ( v11 == *v17 )
            goto LABEL_16;
          v41 = v44;
        }
        LODWORD(v19) = 0;
      }
      v21 = v15 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v22 = (__int64 *)(v14 + 16LL * v21);
      v23 = *v22;
      if ( v12 == *v22 )
      {
LABEL_20:
        v24 = (_QWORD **)v22[1];
        if ( v24 )
        {
          v25 = *v24;
          for ( i = 1; v25; ++i )
            v25 = (_QWORD *)*v25;
LABEL_23:
          if ( i != (_DWORD)v19 )
            return (unsigned int)((_DWORD)v19 - i);
          goto LABEL_41;
        }
      }
      else
      {
        v42 = 1;
        while ( v23 != -4096 )
        {
          v43 = v42 + 1;
          v21 = v15 & (v42 + v21);
          v22 = (__int64 *)(v14 + 16LL * v21);
          v23 = *v22;
          if ( v12 == *v22 )
            goto LABEL_20;
          v42 = v43;
        }
      }
      i = 0;
      goto LABEL_23;
    }
  }
LABEL_41:
  v39 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v40 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
  if ( (_DWORD)v40 != v39 )
    return (unsigned int)(v39 - v40);
  if ( (*((_DWORD *)a3 + 1) & 0x7FFFFFF) == 0 )
    return 0;
  v27 = a4 + 1;
  v28 = 0;
  while ( 1 )
  {
    v29 = (a3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a3 - 1) : &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v30 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    result = sub_D93590(a1, *(_QWORD *)&v30[32 * (unsigned int)v28], *(_QWORD *)&v29[32 * (unsigned int)v28], v27);
    if ( (_DWORD)result )
      break;
    if ( v40 == ++v28 )
      return 0;
  }
  return result;
}
