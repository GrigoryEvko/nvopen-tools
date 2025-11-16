// Function: sub_1E7F580
// Address: 0x1e7f580
//
__int64 __fastcall sub_1E7F580(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // r8
  int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // r10
  int v12; // r9d
  unsigned int v13; // r8d
  __int64 *v14; // rdx
  __int64 v15; // r11
  __int64 v16; // rdx
  unsigned int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // r8
  _QWORD *v20; // rax
  __int64 *v21; // rax
  unsigned int v22; // edx
  __int64 *v23; // r8
  unsigned int v24; // r9d
  __int64 *v25; // rcx
  int v26; // edx
  int v27; // ebx
  int v28; // eax
  int v29; // r11d

  v3 = *a1;
  v6 = *(_QWORD *)v3 + 88LL * *(int *)(a3 + 48);
  result = *(unsigned __int8 *)(v3 + 128);
  if ( (_BYTE)result )
  {
    if ( *(_DWORD *)(v6 + 28) != -1 )
      return 0;
  }
  else if ( *(_DWORD *)(v6 + 24) != -1 )
  {
    return result;
  }
  if ( *(_BYTE *)(a2 + 8) )
  {
    v8 = *(_QWORD *)(v3 + 120);
    v9 = *(_DWORD *)(v8 + 256);
    if ( v9 )
    {
      v10 = *(_QWORD *)a2;
      v11 = *(_QWORD *)(v8 + 240);
      v12 = v9 - 1;
      v13 = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( v10 == *v14 )
      {
LABEL_6:
        v16 = v14[1];
        if ( v16 )
        {
          if ( (_BYTE)result )
            v10 = a3;
          result = 0;
          if ( v10 == **(_QWORD **)(v16 + 32) )
            return result;
          v17 = v12 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v18 = (__int64 *)(v11 + 16LL * v17);
          v19 = *v18;
          if ( a3 != *v18 )
          {
            v28 = 1;
            while ( v19 != -8 )
            {
              v29 = v28 + 1;
              v17 = v12 & (v28 + v17);
              v18 = (__int64 *)(v11 + 16LL * v17);
              v19 = *v18;
              if ( a3 == *v18 )
                goto LABEL_11;
              v28 = v29;
            }
            return 0;
          }
LABEL_11:
          v20 = (_QWORD *)v18[1];
          if ( (_QWORD *)v16 != v20 )
          {
            while ( v20 )
            {
              v20 = (_QWORD *)*v20;
              if ( (_QWORD *)v16 == v20 )
                goto LABEL_14;
            }
            return 0;
          }
        }
      }
      else
      {
        v26 = 1;
        while ( v15 != -8 )
        {
          v27 = v26 + 1;
          v13 = v12 & (v26 + v13);
          v14 = (__int64 *)(v11 + 16LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
            goto LABEL_6;
          v26 = v27;
        }
      }
    }
  }
LABEL_14:
  v21 = *(__int64 **)(v3 + 24);
  if ( *(__int64 **)(v3 + 32) != v21 )
    goto LABEL_15;
  v23 = &v21[*(unsigned int *)(v3 + 44)];
  v24 = *(_DWORD *)(v3 + 44);
  if ( v21 != v23 )
  {
    v25 = 0;
    while ( a3 != *v21 )
    {
      if ( *v21 == -2 )
        v25 = v21;
      if ( v23 == ++v21 )
      {
        if ( !v25 )
          goto LABEL_33;
        *v25 = a3;
        --*(_DWORD *)(v3 + 48);
        ++*(_QWORD *)(v3 + 16);
        return 1;
      }
    }
    return 0;
  }
LABEL_33:
  if ( v24 < *(_DWORD *)(v3 + 40) )
  {
    *(_DWORD *)(v3 + 44) = v24 + 1;
    *v23 = a3;
    ++*(_QWORD *)(v3 + 16);
    return 1;
  }
  else
  {
LABEL_15:
    sub_16CCBA0(v3 + 16, a3);
    return v22;
  }
}
