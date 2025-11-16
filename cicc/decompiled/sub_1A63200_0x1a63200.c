// Function: sub_1A63200
// Address: 0x1a63200
//
__int64 __fastcall sub_1A63200(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v8; // r15
  int v10; // edx
  int v11; // edx
  __int64 v12; // rcx
  unsigned int v13; // esi
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // rdi
  unsigned int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // r9
  bool v21; // al
  __int64 v22; // r12
  int v23; // eax
  _QWORD *v24; // rcx
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // rbx
  int v28; // eax
  int v29; // eax
  int v30; // r8d
  int v31; // r10d

  if ( *(_QWORD *)(a1 + 40) != sub_157F120(a2) )
  {
    v8 = sub_15F2ED0(a1);
    if ( v8 || !sub_15CC8F0(a3, *(_QWORD *)(a1 + 40), a2) )
      return 0;
    v10 = *(_DWORD *)(a4 + 24);
    if ( v10 )
    {
      v11 = v10 - 1;
      v12 = *(_QWORD *)(a4 + 8);
      v13 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = (__int64 *)(v12 + 16LL * (v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
      v15 = *v14;
      if ( a2 == *v14 )
      {
LABEL_7:
        v16 = v14[1];
        v8 = v16 != 0;
      }
      else
      {
        v28 = 1;
        while ( v15 != -8 )
        {
          v30 = v28 + 1;
          v13 = v11 & (v28 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( a2 == *v14 )
            goto LABEL_7;
          v28 = v30;
        }
        v16 = 0;
      }
      v17 = *(_QWORD *)(a1 + 40);
      v18 = v11 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v19 = (__int64 *)(v12 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == v17 )
      {
LABEL_9:
        v21 = v19[1] != v16;
      }
      else
      {
        v29 = 1;
        while ( v20 != -8 )
        {
          v31 = v29 + 1;
          v18 = v11 & (v29 + v18);
          v19 = (__int64 *)(v12 + 16LL * v18);
          v20 = *v19;
          if ( *v19 == v17 )
            goto LABEL_9;
          v29 = v31;
        }
        v21 = v8;
      }
      if ( v21 && v8 )
        return 0;
    }
  }
  v22 = *(_QWORD *)(a1 + 8);
  if ( v22 )
  {
    while ( 1 )
    {
      v26 = sub_1648700(v22);
      v27 = v26;
      if ( *((_BYTE *)v26 + 16) == 77 )
      {
        v23 = sub_1648720(v22);
        v24 = (*((_BYTE *)v27 + 23) & 0x40) != 0 ? (_QWORD *)*(v27 - 1) : &v27[-3 * (*((_DWORD *)v27 + 5) & 0xFFFFFFF)];
        v25 = v24[3 * *((unsigned int *)v27 + 14) + 1 + v23];
      }
      else
      {
        v25 = v26[5];
      }
      if ( !sub_15CC8F0(a3, a2, v25) )
        break;
      v22 = *(_QWORD *)(v22 + 8);
      if ( !v22 )
        return 1;
    }
    return 0;
  }
  return 1;
}
